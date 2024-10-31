
import os
os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import AutoTokenizer,AutoModel,AutoModelForSequenceClassification,BertConfig,PreTrainedModel
# import peft
# from peft import LoraConfig, get_peft_model, PeftModel,IA3Config
import random
import json
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import time

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



# # 模型


class Siamese(PreTrainedModel):
    def __init__(self,config,code_bert='microsoft/codebert-base'):
        super().__init__(config)
        # 隐藏层 size
        self.hidden_size = config.hidden_size
        # 自然语言 bert
        nmodel = code_bert
        # 编程语言 bert
        # pmodel = code_bert
        # 自然语言 词元化器 模型
        self.ntokenizer = AutoTokenizer.from_pretrained(nmodel)
        self.nbert = AutoModel.from_pretrained(nmodel)
        # 编程语言 词元化器 模型
        self.ptokenizer = self.ntokenizer
        self.pbert =  self.nbert
        
        # 平均池化层
        self.npool = torch.nn.AdaptiveAvgPool2d((1, self.hidden_size))
        self.ppool = torch.nn.AdaptiveAvgPool2d((1, self.hidden_size))
        
        # 处理拼接后的线性层
        self.dense = torch.nn.Linear(self.hidden_size*3,self.hidden_size)
        # 随机置零
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # softmax
#         self.softmax = torch.nn.Softmax(dim=1)
        # 分类线性层
        self.output = torch.nn.Linear(self.hidden_size,2)
        
    
    def forward(self,n_ids,n_attention_mask,p_ids,p_attention_mask):
        n_hidden = self.nbert(n_ids,n_attention_mask)[0]
        p_hidden = self.pbert(p_ids,p_attention_mask)[0]
        # 平均池化
        n_pooler = self.npool(n_hidden).view(-1,self.hidden_size)
        p_pooler = self.ppool(p_hidden).view(-1,self.hidden_size)
        # 拼接
        gap_hidden = torch.abs(p_pooler-n_pooler)
        concated_hidden = torch.cat((n_pooler,p_pooler),1)
        concated_hidden = torch.cat((concated_hidden,gap_hidden),1)
        # 分类
        x= self.dropout(concated_hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)#.squeeze(1)
#         x = self.softmax(x)
        return x


class Single(PreTrainedModel):
    def __init__(self, config, code_bert='microsoft/codebert-base'):
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(code_bert)
        self.bert = AutoModel.from_pretrained(code_bert,trust_remote_code=True)
        self.pool = torch.nn.AdaptiveAvgPool2d((1, self.hidden_size))
        # 处理拼接后的线性层
        self.dense = torch.nn.Linear(self.hidden_size,self.hidden_size)
        # 随机置零
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # 分类线性层
        self.output = torch.nn.Linear(self.hidden_size,2)
    def forward(self,ids,attention_mask,type_ids):
        # 输入到bert中
        hidden = self.bert(input_ids=ids,attention_mask=attention_mask,token_type_ids=type_ids)[0]
        # 平均池化
        pooler = self.pool(hidden).view((-1,self.hidden_size))
        # 分类
        # x= self.dropout(pooler)
        # print(hidden.shape, pooler.shape)
        x = self.dense(pooler)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
#         x = self.sigmoid(x)
        return x



# # 数据集


# 创建数据集类
class MyDataset(Dataset):
    def __init__(self, dataframe, tokenizer,flag='divide' ,max_length=512):
        assert flag in ['divide','merge']
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.flag = flag

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        nl = self.dataframe['source_text'].iloc[idx]
        pl = self.dataframe['target_text'].iloc[idx]
        # print(type(pl),pl)
        label = self.dataframe['label'].iloc[idx]
        # nl_id = self.dataframe['source_id'].iloc[idx]
        # pl_id = self.dataframe['target_id'].iloc[idx]
        nl_id = nl
        pl_id = pl
        # 词元化文本数据
        if self.flag=='divide':
            nl = self.tokenizer(nl, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            pl = self.tokenizer(pl, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
            return {
                'n_ids': nl['input_ids'].squeeze(),
                'n_attention_mask': nl['attention_mask'].squeeze(),
                'p_ids': pl['input_ids'].squeeze(),
                'p_attention_mask': pl['attention_mask'].squeeze(),
                'label':torch.tensor(label),
                'nl_id':nl,
                'pl_id':pl
            }
        else:
            nl_pl = nl+self.tokenizer.sep_token+pl
            # nl_pl = self.tokenizer.cls_token+nl+self.tokenizer.sep_token+pl+self.tokenizer.eos_token
            # print(nl_pl)
            nl_pl = self.tokenizer(nl_pl,padding='max_length', truncation=True, max_length=self.max_length, return_token_type_ids=True,return_tensors='pt')
            return {
                'ids':nl_pl['input_ids'].squeeze(),
                'type_ids':nl_pl['token_type_ids'].squeeze(),
                'attention_mask':nl_pl['attention_mask'].squeeze(),
                'label':torch.tensor(label),
                'nl_id':nl_id,
                'pl_id':pl_id
            }


# # 训练参数


class Args:
    def __init__(self,seed,enhance_llm) -> None:
        self.batch_size = 4
        self.lr = 1e-5
        self.epochs = 20
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.momentum = 0.9
        self.gamma = 0.1
        self.k = 10 # K则交叉验证
        self.patience = 5 # 早停类的patience
        self.min_delta = 0 # 早停类的min_delta
        # 随机种子
        self.seed = seed
        # 源数据所在文件夹
        self.raw_data_path = 'myTraceBERT/datasets/raw/'
        # 增强数据所在的文件夹
        self.enhance_data_path = f'myTraceBERT/datasets/{enhance_llm}/'
        # encoder模型名称或路径base
        # self.encoder_base = './models/codebert_base'
        # encoder 分词器名称或路径
        # self.tokenizer_name = './models/codebert_base'
        # 模型保存路径  模型预测保存路径
        self.output_path = f'myTraceBERT/outputs/{self.seed}/'
                

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        # 梯度累计次数
        self.accumulation_steps = 2
        # 模型名称_编码器_数据集_增强数据集_用于数据增强的模型
        # 训练函数的参数要包括模型名称_编码器_数据集名称_数据集增强的方式_用于数据增强的模型
# args = Args()
# set_seed(args.seed)
# args.device


# # 训练函数


def train_siamese(df,data_name,model_name,encoder_path,encoder_name,args,split=[0.8,0.1,0.1],enhance_type="none",enhance_llm="none",data_enhance=None,criterion = nn.CrossEntropyLoss()):
    """
    df : 未划分的数据集(必填)
    data_name : 数据集名称*(必填)
    args : 训练参数
    split : 划分数据集的比例
    enhance_type : 数据增强的方式 包括 none req code req(example) code(example)
    enhance_llm : 用于数据增强的模型名称*
    data_enhance : 数据增强的数据集
    model_name : 模型名称*(必填)
    encoder_path : 编码器模型路径*(必填)
    encoder_name : 编码器名称*(必填)
    criterion : 损失函数
    """
    assert enhance_type in ["none","req","code","req(example)", "code(example)"]
    assert encoder_name in ["CodeBERT","Jina"]
    assert sum(split) == 1

    if enhance_type!="none" and enhance_llm=="none":
        print("使用数据增强但用于数据增强的模型为空，请重新输入!")
        assert enhance_llm!="none"
    print(f"-----------------{model_name}-{encoder_name}-{data_name}-{enhance_type}-{enhance_llm}--------------------")
    model = Siamese(BertConfig(),code_bert=encoder_path)

    if encoder_name == "CodeBERT":
        max_length = 512
    if encoder_name == "Jina":
        max_length = 1024
        # model.load_state_dict(torch.load(f'{encoder_path}/Single-Jina.pth'))
    softmax = torch.nn.Softmax()
    # if model_name == 'siamese':
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)
    model.to(args.device)

    result_record = pd.read_excel("myTraceBERT/outputs/result_record.xlsx")
    

    best_val_acc = 0
    best_loss = 100
    df_1 = df[df['label'] == 1]
    df_0 = df[df['label'] == 0]
    num_1 = len(df_1)
    num_0 = len(df_0)
    train_data = pd.concat([df_1.iloc[:int(split[0]*num_1)],df_0.iloc[:int(split[0]*num_0)]])
    valid_data = pd.concat([df_1.iloc[int(split[0]*num_1):int((split[0]+split[1])*num_1)],df_0.iloc[int(split[0]*num_0):int((split[0]+split[1])*num_0)]])
    test_data = pd.concat([df_1.iloc[int((split[0]+split[1])*num_1):],df_0.iloc[int((split[0]+split[1])*num_0):]])
    if enhance_type!="none":
        enhance = data_enhance[['source_text','target_text','label']]#.sample(frac=1)
        train_data = pd.concat([train_data,enhance]).dropna().sample(frac=1)
    print(f"train_num:{len(train_data)},valid_num:{len(valid_data)},test_num:{len(test_data)}")
    tokenizer=AutoTokenizer.from_pretrained(encoder_path)
    train_dataset = MyDataset(train_data,tokenizer,max_length=max_length)
    valid_dataset = MyDataset(valid_data,tokenizer,max_length=max_length)
    test_dataset = MyDataset(test_data,tokenizer,max_length=max_length)
    # 数据加载器
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,pin_memory=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True,pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,pin_memory=True)


    # 损失函数 优化器

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    # 早停
    # earlyStop = EarlyStopping(args.patience,args.min_delta)
    for epoch in range(args.epochs):
        # ------------------train----------------------------------
        model.train()
        acc = 0
        nums =0
        train_loss_list = []
        for batch in train_data_loader:
            n_ids = batch['n_ids'].to(args.device)
            n_attention_mask = batch['n_attention_mask'].to(args.device)
            p_ids = batch['p_ids'].to(args.device)
            p_attention_mask = batch['p_attention_mask'].to(args.device)
            label = batch['label'].to(args.device)

            outputs = model(n_ids,n_attention_mask,p_ids,p_attention_mask)

            optimizer.zero_grad()
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            outputs = softmax(outputs)
            for i in range(len(outputs)):
                if torch.argmax(outputs[i])==label[i]:
                    acc+=1
            nums+=len(outputs)
            train_loss_list.append(loss.item())
        print('Train:Epoch {}, Loss:{},Acc:{}'.format(epoch,sum(train_loss_list)/len(train_loss_list),acc/nums))
        # -------------------------Valid--------------------------------------
        with torch.no_grad():
            model.eval()
            val_loss_list = []
            acc = 0
            nums =0
            for batch in valid_data_loader:
                n_ids = batch['n_ids'].to(args.device)
                n_attention_mask = batch['n_attention_mask'].to(args.device)
                p_ids = batch['p_ids'].to(args.device)
                p_attention_mask = batch['p_attention_mask'].to(args.device)
                label = batch['label'].to(args.device)

                outputs = model(n_ids,n_attention_mask,p_ids,p_attention_mask)
                loss = criterion(outputs, label)
                val_loss_list.append(loss.item())
                outputs = softmax(outputs)
                for i in range(len(outputs)):
                    if torch.argmax(outputs[i])==label[i]:
                        acc+=1

                nums+=len(outputs)
            print('Valid:Epoch {}, Loss:{},Acc:{}'.format(epoch,sum(val_loss_list)/len(val_loss_list),acc/nums))
            if best_val_acc < acc/nums or (best_val_acc==acc/nums and best_loss > sum(val_loss_list)/len(val_loss_list)) :
                best_val_acc = acc/nums
                best_loss = sum(val_loss_list)/len(val_loss_list)
                # 保存模型
                print(epoch,'best_val_acc',acc/nums)
                torch.save(model.state_dict(), f'{args.output_path}{model_name}-{encoder_name}-{data_name}-{enhance_type}-{enhance_llm}-{args.seed}.pth')

    # epoch结束
    # -------------------------Test--------------------------------------
    model.load_state_dict(torch.load(f'{args.output_path}{model_name}-{encoder_name}-{data_name}-{enhance_type}-{enhance_llm}-{args.seed}.pth'))
    model.eval()
    list_for_map = []
    pred_result = []
    truth_result = []
    record = {"label":[],"pred":[],"score":[]}
    for batch in test_data_loader:
        n_ids = batch['n_ids'].to(args.device)
        n_attention_mask = batch['n_attention_mask'].to(args.device)
        p_ids = batch['p_ids'].to(args.device)
        p_attention_mask = batch['p_attention_mask'].to(args.device)
        label = batch['label'].to(args.device)


        outputs = model(n_ids,n_attention_mask,p_ids,p_attention_mask)
        outputs = softmax(outputs)
        for i in range(len(outputs)):
            pred_result.append(torch.argmax(outputs[i]).item())
            truth_result.append(label[i].item())
            record["label"].append(label[i].item())
            record["pred"].append(torch.argmax(outputs[i]).item())
            record["score"].append(torch.max(outputs[i]).item())

    # 计算指标
    result = {}
    result["model_name"] = model_name
    result["encoder_name"] = encoder_name
    result["data_name"] = data_name
    result["enhance_type"] = enhance_type
    result["enhance_llm"] = enhance_llm
    result['seed'] = args.seed
    result['Accuracy'] = accuracy_score(truth_result,pred_result)
    result['Precision'] = precision_score(truth_result,pred_result)
    result['Recall'] = recall_score(truth_result,pred_result)
    result['F1'] = f1_score(truth_result,pred_result)
    print(json.dumps(result, indent=4, ensure_ascii=False))
    # 保存模型
    # torch.save(model.state_dict(), f'{data_name}-{model_name}-{fold}-{f1_score(truth_result,pred_result)}.pth')
    # 删除模型
    os.remove(f'{args.output_path}{model_name}-{encoder_name}-{data_name}-{enhance_type}-{enhance_llm}-{args.seed}.pth')
    del list_for_map,pred_result,truth_result,train_data,valid_data,test_data,train_dataset,valid_dataset,test_dataset,train_data_loader,valid_data_loader,test_data_loader

    for _ in range(10):
        torch.cuda.empty_cache()
    time.sleep(5)
    del model
    pd.DataFrame(record).to_csv(args.output_path+f'{model_name}-{encoder_name}-{data_name}-{enhance_type}-{enhance_llm}-{args.seed}-record.csv')
    result_record = pd.concat([result_record,pd.DataFrame(result,index=[0])],ignore_index=True)
    result_record.to_excel("./outputs/result_record.xlsx",index=False)



def train_single(df,data_name,model_name,encoder_path,encoder_name,args,split=[0.8,0.1,0.1],enhance_type="none",enhance_llm="none",data_enhance=None,criterion = nn.CrossEntropyLoss()):
    """
    df : 未划分的数据集(必填)
    data_name : 数据集名称*(必填)
    args : 训练参数
    split : 划分数据集的比例
    enhance_type : 数据增强的方式 包括 none req code req(example) code(example)
    enhance_llm : 用于数据增强的模型名称*
    data_enhance : 数据增强的数据集
    model_name : 模型名称*(必填)
    encoder_path : 编码器模型路径*(必填)
    encoder_name : 编码器名称*(必填)
    criterion : 损失函数
    """
    assert enhance_type in ["none","req","code","req(example)", "code(example)"]
    assert encoder_name in ["CodeBERT","Jina","Jina1024","Jina1536"]
    assert sum(split) == 1

    if enhance_type!="none" and enhance_llm=="none":
        print("使用数据增强但用于数据增强的模型为空，请重新输入!")
        assert enhance_llm!="none"
    print(f"-----------------{model_name}-{encoder_name}-{data_name}-{enhance_type}-{enhance_llm}--------------------")
    model = Single(BertConfig(),code_bert=encoder_path)

    if encoder_name == "CodeBERT":
        max_length = 512
    if encoder_name == "Jina":
        max_length = 512
    if encoder_name == "Jina1024":
        max_length = 1024
    if encoder_name == "Jina1536":
        max_length = 1536
        # model.load_state_dict(torch.load(f'{encoder_path}/Single-Jina.pth'))
    softmax = torch.nn.Softmax()
    
    model = Single(BertConfig(),code_bert=encoder_path)
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)
    model.to(args.device)
    

    result_record = pd.read_excel("myTraceBERT/outputs/result_record.xlsx")
    
    

    best_val_acc = 0
    best_loss = 100
    # 划分数据集
    df_1 = df[df['label'] == 1]
    df_0 = df[df['label'] == 0]
    num_1 = len(df_1)
    num_0 = len(df_0)
    train_data = pd.concat([df_1.iloc[:int(split[0]*num_1)],df_0.iloc[:int(split[0]*num_0)]])
    valid_data = pd.concat([df_1.iloc[int(split[0]*num_1):int((split[0]+split[1])*num_1)],df_0.iloc[int(split[0]*num_0):int((split[0]+split[1])*num_0)]])
    test_data = pd.concat([df_1.iloc[int((split[0]+split[1])*num_1):],df_0.iloc[int((split[0]+split[1])*num_0):]])
    if enhance_type!="none":
        enhance = data_enhance[['source_text','target_text','label']]#.sample(frac=1)
        train_data = pd.concat([train_data,enhance]).dropna().sample(frac=1)
    print(f"train_num:{len(train_data)},valid_num:{len(valid_data)},test_num:{len(test_data)}")
    tokenizer=AutoTokenizer.from_pretrained(encoder_path)
    train_dataset = MyDataset(train_data,tokenizer,flag='merge',max_length=max_length)
    valid_dataset = MyDataset(valid_data,tokenizer,flag='merge',max_length=max_length)
    test_dataset = MyDataset(test_data,tokenizer,flag='merge',max_length=max_length)
    # 数据加载器
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,pin_memory=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True,pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,pin_memory=True)



    # 损失函数 优化器

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

    for epoch in range(args.epochs):
        # ------------------train----------------------------------
        model.train()
        acc = 0
        nums =0
        train_loss_list = []
        for i,batch in enumerate(train_data_loader):
            n_ids = batch['ids'].to(args.device)
            n_attention_mask = batch['attention_mask'].to(args.device)

            token_ids = batch['type_ids'].to(args.device)
            label = batch['label'].to(args.device)

            outputs = model(n_ids,n_attention_mask,token_ids)

            # optimizer.zero_grad()
            loss = criterion(outputs, label)
            loss.backward()
            # optimizer.step()
            # 梯度累积
            if (i+1)% args.accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            outputs = softmax(outputs)
            for i in range(len(outputs)):
                if torch.argmax(outputs[i])==label[i]:
                    acc+=1
            nums+=len(outputs)
            train_loss_list.append(loss.item())
        print('Train:Epoch {}, Loss: {},Acc:{}'.format(epoch,sum(train_loss_list)/len(train_loss_list),acc/nums))
        # -------------------------Valid--------------------------------------
        with torch.no_grad():
            model.eval()
            val_loss_list = []
            acc = 0
            nums =0
            for batch in valid_data_loader:
                n_ids = batch['ids'].to(args.device)
                n_attention_mask = batch['attention_mask'].to(args.device)

                token_ids = batch['type_ids'].to(args.device)
                label = batch['label'].to(args.device)

                outputs = model(n_ids,n_attention_mask,token_ids)
                loss = criterion(outputs, label)
                val_loss_list.append(loss.item())
                outputs = softmax(outputs)
                for i in range(len(outputs)):
                    if torch.argmax(outputs[i])==label[i]:
                        acc+=1

                nums+=len(outputs)
            print('Valid:Epoch {}, Loss:{},Acc:{}'.format(epoch,sum(val_loss_list)/len(val_loss_list),acc/nums))
            if best_val_acc < acc/nums or (best_val_acc==acc/nums and best_loss > sum(val_loss_list)/len(val_loss_list)) :
                best_val_acc = acc/nums
                best_loss = sum(val_loss_list)/len(val_loss_list)
                print(epoch,'best_val_acc:',best_val_acc)             
                torch.save(model.state_dict(), f'{args.output_path}{model_name}-{encoder_name}-{data_name}-{enhance_type}-{enhance_llm}-{args.seed}.pth')

    # epoch结束
    # -------------------------Test--------------------------------------
    del optimizer,loss
    model.load_state_dict(torch.load(f'{args.output_path}{model_name}-{encoder_name}-{data_name}-{enhance_type}-{enhance_llm}-{args.seed}.pth'))

    model.eval()
    list_for_map = []
    pred_result = []
    truth_result = []
    record = {"label":[],"pred":[],"score":[]}
    for batch in test_data_loader:
        n_ids = batch['ids'].to(args.device)
        n_attention_mask = batch['attention_mask'].to(args.device)

        token_ids = batch['type_ids'].to(args.device)
        label = batch['label'].to(args.device)


        outputs = model(n_ids,n_attention_mask,token_ids)
        outputs = softmax(outputs)
        for i in range(len(outputs)):
            pred_result.append(torch.argmax(outputs[i]).item())
            truth_result.append(label[i].item())
            record["label"].append(label[i].item())
            record["pred"].append(torch.argmax(outputs[i]).item())
            record["score"].append(torch.max(outputs[i]).item())
   
    # 计算指标
    result = {}
    result["model_name"] = model_name
    result["encoder_name"] = encoder_name
    result["data_name"] = data_name
    result["enhance_type"] = enhance_type
    result["enhance_llm"] = enhance_llm
    result['seed'] = args.seed
    result['Accuracy'] = accuracy_score(truth_result,pred_result)
    result['Precision'] = precision_score(truth_result,pred_result)
    result['Recall'] = recall_score(truth_result,pred_result)
    result['F1'] = f1_score(truth_result,pred_result)
    print(json.dumps(result, indent=4, ensure_ascii=False))
    # result['MAP'].append(MAP_at_K(pd.DataFrame(list_for_map),k=3))
    # print(f1_score(truth_result,pred_result))
    # 保存模型
    # torch.save(model.state_dict(), f'{data_name}-{model_name}-{fold}-{f1_score(truth_result,pred_result)}.pth')
    # 删除模型
    os.remove(f'{args.output_path}{model_name}-{encoder_name}-{data_name}-{enhance_type}-{enhance_llm}-{args.seed}.pth')
    del list_for_map,pred_result,truth_result,train_data,valid_data,test_data,train_dataset,valid_dataset,test_dataset,train_data_loader,valid_data_loader,test_data_loader

    for _ in range(10):
        torch.cuda.empty_cache()
        
    time.sleep(5)
    del model
    pd.DataFrame(record).to_csv(args.output_path+f'{model_name}-{encoder_name}-{data_name}-{enhance_type}-{enhance_llm}-{args.seed}-record.csv')
    # result_record = result_record.append(result,ignore_index=True)
    result_record = pd.concat([result_record,pd.DataFrame(result,index=[0])],ignore_index=True)
    result_record.to_excel("myTraceBERT/outputs/result_record.xlsx",index=False)



# # 训练

# ## jina 1536


for s in range(2014,2019):
    args = Args(seed=s,enhance_llm="gemini") ###
    print(vars(args))
    set_seed(args.seed)
    print(args.device,args.seed,args.enhance_data_path)
    # 训练
    for dn in ["EBT","RETRO","eTOUR","iTrust"]:
        df = pd.read_csv(f'{args.raw_data_path}/full_{dn}_link.csv')
        model_name = "Single"
        encoder_name = "Jina1536"
        encoder_path = "myTraceBERT/models/jina"
        enhance_llm="Gemini" ###
        # 原数据
        # train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args)
        # 无sample-需求
        train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="req",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_gen_requirement_{dn}.xlsx"))
        # 无sample-代码
        train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="code",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_gen_Code_{dn}.xlsx"))
        # sample-需求
        train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="req(example)",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_example_gen_requirement_{dn}.xlsx"))
        # sample-代码
        train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="code(example)",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_example_gen_Code_{dn}.xlsx"))



for s in range(2014,2019):
    args = Args(seed=s,enhance_llm="gpt3.5") ###
    print(vars(args))
    set_seed(args.seed)
    print(args.device,args.seed,args.enhance_data_path)
    # 训练
    for dn in ["EBT","RETRO","eTOUR","iTrust"]:
        df = pd.read_csv(f'{args.raw_data_path}/full_{dn}_link.csv')
        model_name = "Single"
        encoder_name = "Jina1536"
        encoder_path = "myTraceBERT/models/jina"
        enhance_llm="GPT3.5" ###
        # 原数据
        # train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args)
        # 无sample-需求
        train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="req",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_gen_requirement_{dn}.xlsx"))
        # 无sample-代码
        train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="code",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_gen_Code_{dn}.xlsx"))
        # sample-需求
        train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="req(example)",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_example_gen_requirement_{dn}.xlsx"))
        # sample-代码
        train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="code(example)",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_example_gen_Code_{dn}.xlsx"))



for s in range(2014,2019):
    args = Args(seed=s,enhance_llm="claude3") ###
    print(vars(args))
    set_seed(args.seed)
    print(args.device,args.seed,args.enhance_data_path)
    # 训练
    for dn in ["EBT","RETRO","eTOUR","iTrust"]:
        df = pd.read_csv(f'{args.raw_data_path}/full_{dn}_link.csv')
        model_name = "Single"
        encoder_name = "Jina1536"
        encoder_path = "myTraceBERT/models/jina"
        enhance_llm="Claude3" ###
        # 原数据
        # train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args)
        # 无sample-需求
        train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="req",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_gen_requirement_{dn}.xlsx"))
        # 无sample-代码
        train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="code",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_gen_Code_{dn}.xlsx"))
        # sample-需求
        train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="req(example)",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_example_gen_requirement_{dn}.xlsx"))
        # sample-代码
        train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="code(example)",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_example_gen_Code_{dn}.xlsx"))



for s in range(2014,2019):
    args = Args(seed=s,enhance_llm="gpt4o") ###
    print(vars(args))
    set_seed(args.seed)
    print(args.device,args.seed,args.enhance_data_path)
    # 训练
    for dn in ["EBT","RETRO","eTOUR","iTrust"]:
        df = pd.read_csv(f'{args.raw_data_path}/full_{dn}_link.csv')
        model_name = "Single"
        encoder_name = "Jina1536"
        encoder_path = "myTraceBERT/models/jina"
        enhance_llm="GPT-4o" ###
        # 原数据
        # train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args)
        # 无sample-需求
        train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="req",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_gen_requirement_{dn}.xlsx"))
        # 无sample-代码
        train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="code",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_gen_Code_{dn}.xlsx"))
        # sample-需求
        train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="req(example)",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_example_gen_requirement_{dn}.xlsx"))
        # sample-代码
        train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="code(example)",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_example_gen_Code_{dn}.xlsx"))




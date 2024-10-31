
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize



import gensim
from gensim.models import Word2Vec
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import string
import re
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import json
import time
import os
import random
# 下载停用词列表
# nltk.download('stopwords')

# 定义停用词
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 将非英文字符和非数字字符替换为空格
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = text.replace("  "," ")
    text = text.lower()
    
    
    # 将文本拆分为单词
    words = word_tokenize(text)
    
    # 去除停用词
    filtered_words = [word for word in words if word not in stop_words]
    
    
    return filtered_words
preprocess_text("Deep learning is fascinating")



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False



# # RNN 模型


class RNN(nn.Module):
    def __init__(self,model_name,input_size,hidden_size=30,dropout=0):
        super().__init__()
        self.hidden_size = hidden_size
        self.model_name = model_name
        if model_name == 'gru':
            self.rnn = nn.GRU(input_size,hidden_size,batch_first=True,dropout=dropout)
        if model_name == "lstm":
            self.rnn = nn.LSTM(input_size,hidden_size,batch_first=True,dropout=dropout)
        if model_name == 'bi_gru':
            self.rnn = nn.GRU(input_size,hidden_size,batch_first=True,bidirectional=True,dropout=dropout)
        if model_name == "bi_lstm":
            self.rnn = nn.LSTM(input_size,hidden_size,batch_first=True,bidirectional=True,dropout=dropout)
            
    def forward(self,x):

        if "bi" in self.model_name:
            h0 = torch.zeros(2,x.size(0),self.hidden_size).to(args.device)
            c0 = torch.zeros(2,x.size(0),self.hidden_size).to(args.device)
        else:
            h0 = torch.zeros(1,x.size(0),self.hidden_size).to(args.device)
            c0 = torch.zeros(1,x.size(0),self.hidden_size).to(args.device)
        if "lstm" in self.model_name:
            output,_ = self.rnn(x,(h0,c0))
        else:
            output,_ = self.rnn(x,h0)
#         print("output",output[:,-1,:].shape)
        return output[:,-1,:]
class RNNTracer(nn.Module):
    def __init__(self,model_name,input_size,args,hidden_size=30,dropout=0,):
        super().__init__()
        self.hidden_size = hidden_size
        self.model_name = model_name
        # self.nl_rnn = RNN(model_name,input_size).to(args.device)
        # self.pl_rnn = RNN(model_name,input_size).to(args.device)
        self.rnn = RNN(model_name,input_size).to(args.device)
        self.dp = dropout
        self.args = args
        # 处理拼接后的线性层
        if "bi" in model_name:
            self.dense = nn.Linear(self.hidden_size*2,self.hidden_size).to(args.device) # *4
        else:
            self.dense = nn.Linear(self.hidden_size*1,self.hidden_size).to(args.device) # *2
        # 随机置零
        self.dropout = nn.Dropout(self.dp)
        # 分类线性层
        self.output = nn.Linear(self.hidden_size,2).to(args.device)
    def forward(self,nl,pl):
        # nl_hidden = self.nl_rnn(nl).to(self.args.device)
        # pl_hidden = self.pl_rnn(pl).to(self.args.device)
        # concated_hidden = torch.concat((nl_hidden,pl_hidden),1)
        sep_token = torch.tensor(np.zeros(args.embedding_dim), dtype=torch.float32).to(args.device)
        hidden = self.rnn(nl+sep_token+pl).to(self.args.device)
        # 分类
        x = self.dropout(hidden)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
        return x


# # 训练参数


class Args:
    def __init__(self,seed) -> None:
        self.batch_size = 1024
        self.lr = 1e-2
        self.epochs = 200
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 词嵌入维度
        self.embedding_dim = 50
        # RNN 维度
        self.rnn_hidden_size = 30
        # 合并层维度
        self.intg_hidden_size = 10
        # dropout 
        self.dp = 0.1
        # 文本最大长度
        self.max_len = 512
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
#         self.enhance_data_path = f'myTraceBERT/datasets/{enhance_llm}/'
        # encoder模型名称或路径base
        # self.encoder_base = './models/codebert_base'
        # encoder 分词器名称或路径
        # self.tokenizer_name = './models/codebert_base'
        # 模型保存路径  模型预测保存路径
        self.output_path = f'myTraceBERT/outputs/{self.seed}/'
                

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
#         # 梯度累计次数
#         self.accumulation_steps = 2
        # 模型名称_编码器_数据集_增强数据集_用于数据增强的模型
        # 训练函数的参数要包括模型名称_编码器_数据集名称_数据集增强的方式_用于数据增强的模型
args = Args(2024)
# set_seed(args.seed)
# args.device


# 数据集
class MyDataset(Dataset):
    # args
    def __init__(self,df,max_len=128):
        """
        df: 数据DataFRame
        max_len: 文本最大序列长度
        """
        df["source_tokens"] = df["source_text"].apply(preprocess_text)
        df["target_tokens"] = df["target_text"].apply(preprocess_text)
        self.source_tokens = df["source_tokens"].values.tolist()
        self.target_tokens = df["target_tokens"].values.tolist()
        self.labels = df['label'].values.tolist()
        # args
        source_w2v = Word2Vec(self.source_tokens+self.target_tokens, vector_size=args.embedding_dim, window=5, min_count=1, workers=4)
        self.word_vectors = source_w2v.wv
        # args
        self.embedding_dim = source_w2v.vector_size
        # args
        self.max_len = args.max_len
    def __len__(self):
        return len(self.source_tokens)
    def __getitem__(self,idx):
        nl_words = self.source_tokens[idx]
        pl_words = self.target_tokens[idx]
        label = self.labels[idx]

        nl_vec = np.array([self.word_vectors[word] if word in self.word_vectors else np.zeros(self.embedding_dim) for word in nl_words])
        pl_vec = np.array([self.word_vectors[word] if word in self.word_vectors else np.zeros(self.embedding_dim) for word in pl_words])
        if len(nl_words) >= self.max_len:
            nl_vec = nl_vec[:self.max_len]
        else:
            middle = np.zeros([self.max_len,self.embedding_dim])
            middle[:len(nl_words)] = nl_vec
            nl_vec = middle
        if len(pl_words) >= self.max_len:
            pl_vec = pl_vec[:self.max_len]
        else:   
            middle = np.zeros([self.max_len,self.embedding_dim])
            middle[:len(pl_words)] = pl_vec
            pl_vec = middle
            
        nl_vec = torch.tensor(nl_vec, dtype=torch.float32).to(args.device)
        pl_vec = torch.tensor(pl_vec, dtype=torch.float32).to(args.device)
#         print("nl",nl_words)
#         print("pl",pl_words)
#         print("label",label)
        return {
            "nl":nl_vec,
            "pl":pl_vec,
            "label": torch.tensor(label).to(args.device)
        }
# train_data = MyDataset(df)
# train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,pin_memory=True)


# train_data[2]


# for i in range(100):
#     print(train_data[i]["nl"].shape)
# # train_data[0]





# # 嵌入维度50
# vector_size = 50
# rnntrace = RNNTracer(model_name='bi_lstm',input_size=args.embedding_dim)
# # rnn = RNN(model_name='bi_lstm',input_size=vector_size)
# rnntrace(train_data[1]["nl"].unsqueeze(0).to(args.device),train_data[1]["pl"].unsqueeze(0).to(args.device))


# for i in train_loader:
#     print(i)
#     break


# # 训练函数


def train_rnn(df,data_name,model_name,args,split=[0.8,0.1,0.1],enhance_type="none",enhance_llm="none",data_enhance=None,criterion = nn.CrossEntropyLoss()):
    """
    df : 未划分的数据集(必填)
    data_name : 数据集名称*(必填)
    args : 训练参数
    split : 划分数据集的比例
    enhance_type : 数据增强的方式 包括 none req code req(example) code(example)
    enhance_llm : 用于数据增强的模型名称*
    data_enhance : 数据增强的数据集
    model_name : 模型名称*(必填)
    #encoder_path : 编码器模型路径*(必填)
    #encoder_name : 编码器名称*(必填)
    criterion : 损失函数
    """
    assert enhance_type in ["none","req","code","req(example)", "code(example)"]
#     assert encoder_name in ["CodeBERT","Jina","Jina1024"]
    assert sum(split) == 1

    if enhance_type!="none" and enhance_llm=="none":
        print("使用数据增强但用于数据增强的模型为空，请重新输入!")
        assert enhance_llm!="none"
    print(f"-----------------{model_name}-{data_name}-{enhance_type}-{enhance_llm}--------------------")
#     model = Single(BertConfig(),code_bert=encoder_path)
    model = RNNTracer(model_name=model_name,input_size=args.embedding_dim,dropout=args.dp,args=args)

#     if encoder_name == "CodeBERT":
#         max_length = 512
#     if encoder_name == "Jina":
#         max_length = 512
#     if encoder_name == "Jina1024":
#         max_length = 1024
        # model.load_state_dict(torch.load(f'{encoder_path}/Single-Jina.pth'))
    softmax = torch.nn.Softmax()
    
#     model = Single(BertConfig(),code_bert=encoder_path)
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)
    model.to(args.device)
    
    # result = {
    #     'Accuracy':[],
    #     'Precision':[],
    #     'Recall':[],
    #     'F1':[],
    # }
    
    
    
    #=============================================================================================
    result_record = pd.read_excel("myTraceBERT/outputs/result_record.xlsx")
     #=============================================================================================
    
    
    
    
    # fold = "no_K"
#     for fold ,(label_1,label_0) in enumerate(zip(kf.split(df[df['label'] == 1]),kf.split(df[df['label'] == 0]))):
#         print(f'Fold-{fold}')
        # train_1_index
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
#     train_data = pd.concat([df[df['label'] == 1].iloc[label_1[0]],df[df['label'] == 0].iloc[label_0[0]]])#.sample(frac=1)
    if enhance_type!="none":
        enhance = data_enhance[['source_text','target_text','label']]#.sample(frac=1)
        # print(enhance)
        train_data = pd.concat([train_data,enhance]).dropna().sample(frac=1)
    print(f"train_num:{len(train_data)},valid_num:{len(valid_data)},test_num:{len(test_data)}")
#     valid_data = train_data.head(int(len(train_data)*0.1111))
#     train_data = train_data.tail(len(train_data)-int(len(train_data)*0.1111))
#     test_data = pd.concat([df[df['label'] == 1].iloc[label_1[1]],df[df['label'] == 0].iloc[label_0[1]]])#.sample(frac=1)
#     tokenizer=AutoTokenizer.from_pretrained(encoder_path)
    train_dataset = MyDataset(train_data,max_len=args.max_len)
    valid_dataset = MyDataset(valid_data,max_len=args.max_len)
    test_dataset = MyDataset(test_data,max_len=args.max_len)
    # 数据加载器
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,pin_memory=False)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True,pin_memory=False)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,pin_memory=False)

    # 加载模型
    # model.load_state_dict(torch.load('/kaggle/working/Single-base.pth'))

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
        for i,batch in enumerate(train_data_loader):
#             n_ids = batch['ids'].to(args.device)
#             n_attention_mask = batch['attention_mask'].to(args.device)
#             # p_ids = batch['p_ids'].to(args.device)
#             # p_attention_mask = batch['p_attention_mask'].to(args.device)
#             token_ids = batch['type_ids'].to(args.device)
#             label = batch['label'].to(args.device)
            nl = batch['nl'].to(args.device)
#             print("nl:",nl.shape,"\n",nl)
            pl = batch['pl'].to(args.device)
            label = batch['label'].to(args.device)

            outputs = model(nl,pl)
#             print('预测',outputs.shape)

            optimizer.zero_grad()
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()
            # 梯度累积
#             if (i+1)% args.accumulation_steps == 0:
#                 optimizer.step()
#                 optimizer.zero_grad()
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
                nl = batch['nl'].to(args.device)
                pl = batch['pl'].to(args.device)
                label = batch['label'].to(args.device)

                outputs = model(nl,pl)
                loss = criterion(outputs, label)
                val_loss_list.append(loss.item())
                outputs = softmax(outputs)
                for i in range(len(outputs)):
                    if torch.argmax(outputs[i])==label[i]:
                        acc+=1

                nums+=len(outputs)
            print('Valid:Epoch {}, Loss:{},Acc:{}'.format(epoch,sum(val_loss_list)/len(val_loss_list),acc/nums))
            # earlyStop(sum(val_loss_list)/len(val_loss_list),acc/nums)
            if best_val_acc < acc/nums or (best_val_acc==acc/nums and best_loss > sum(val_loss_list)/len(val_loss_list)) :
                best_val_acc = acc/nums
                best_loss = sum(val_loss_list)/len(val_loss_list)
                print(epoch,'best_val_acc:',best_val_acc)
                
                
                 #=============================================================================================
                # torch.save(model.state_dict(), f'{model_name}-{data_name}-{enhance_type}-{enhance_llm}-{args.seed}.pth')
                torch.save(model.state_dict(), f'{args.output_path}{model_name}-{data_name}-{enhance_type}-{enhance_llm}-{args.seed}.pth')
                 #=============================================================================================
                
                
        # 早停
        # if earlyStop.early_stop:

        #     break
    # epoch结束
    # -------------------------Test--------------------------------------
    del optimizer,loss
     #=============================================================================================
    model.load_state_dict(torch.load(f'{args.output_path}{model_name}-{data_name}-{enhance_type}-{enhance_llm}-{args.seed}.pth'))
 #=============================================================================================
    model.eval()
    list_for_map = []
    pred_result = []
    truth_result = []
    record = {"label":[],"pred":[],"score":[]}
    for batch in test_data_loader:
        nl = batch['nl'].to(args.device)
        pl = batch['pl'].to(args.device)
        label = batch['label'].to(args.device)

        outputs = model(nl,pl)
        outputs = softmax(outputs)
        for i in range(len(outputs)):
            pred_result.append(torch.argmax(outputs[i]).item())
            truth_result.append(label[i].item())
            record["label"].append(label[i].item())
            record["pred"].append(torch.argmax(outputs[i]).item())
            record["score"].append(torch.max(outputs[i]).item())
            # print(i,outputs[i])
            # if torch.argmax(outputs[i])==1:

                # list_for_map.append({"nl_id":nl_id[i],"pl_id":pl_id[i],"pred":outputs[i][1].item(),'label':label[i].item()})
    # print(pd.DataFrame(list_for_map))     
    # 计算指标
    result = {}
    result["model_name"] = model_name
#     result["encoder_name"] = encoder_name
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
    #==============================================================
    os.remove(f'{args.output_path}{model_name}-{data_name}-{enhance_type}-{enhance_llm}-{args.seed}.pth')
    # ===================================================
    del list_for_map,pred_result,truth_result,train_data,valid_data,test_data,train_dataset,valid_dataset,test_dataset,train_data_loader,valid_data_loader,test_data_loader

    for _ in range(10):
        torch.cuda.empty_cache()
        
    time.sleep(5)
    del model
    
    
     #=============================================================================================
    # pd.DataFrame(record).to_csv(f'{model_name}-{data_name}-{enhance_type}-{enhance_llm}-{args.seed}-record.csv')
    pd.DataFrame(record).to_csv(args.output_path+f'{model_name}-{data_name}-{enhance_type}-{enhance_llm}-{args.seed}-record.csv')
     #=============================================================================================
    
    
    # result_record = result_record.append(result,ignore_index=True)
    
    
    
     #=============================================================================================
    result_record = pd.concat([result_record,pd.DataFrame(result,index=[0])],ignore_index=True)
    result_record.to_excel("myTraceBERT/outputs/result_record.xlsx",index=False)
     #=============================================================================================



    # pd.DataFrame(result).to_csv(args.output_path+f'{model_name}-{encoder_name}-{data_name}-{enhance_type}-{enhance_llm}-metric.csv')
    # print('Accuracy',sum(result['Accuracy'])/len(result['Accuracy']))
    # print('Precision',sum(result['Precision'])/len(result['Precision']))
    # print('Recall',sum(result['Recall'])/len(result['Recall']))
    # print('F1',sum(result['F1'])/len(result['F1']))
    # print('MAP',sum(result['MAP'])/len(result['MAP']))


# # 训练


type2df = {
    "req":"full_pure_gen_requirement_",
    "code":"full_pure_gen_Code_",
    "req(example)":"full_pure_example_gen_requirement_",
    "code(example)":"full_pure_example_gen_Code_"
}
for seed in range(2014,2019):
    args = Args(seed)
    print(vars(args))
    set_seed(args.seed)
    print(args.device,args.seed)
    for llm in ["claude3","gemini","gpt3.5","gpt4o"]:
        # 训练
        for dn in ["RETRO","eTOUR","iTrust","EBT"]:
            for t in ["req","code","req(example)","code(example)"]:
                for model in ['lstm','gru','bi_lstm','bi_gru']:
                    print(f"-----{dn}--{model}--{seed}-----------")
                    df = pd.read_csv(f"myTraceBERT/datasets/raw/full_{dn}_link.csv")
                    enhance_df = pd.read_excel(f"myTraceBERT/datasets/{llm}/{type2df[t]}{dn}.xlsx")
                    train_rnn(df=df,data_name=dn,model_name=model,args=args,enhance_type=t,enhance_llm=llm,data_enhance=enhance_df)















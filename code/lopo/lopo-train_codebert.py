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



# # Model


class Siamese(PreTrainedModel):
    def __init__(self,config,code_bert='microsoft/codebert-base'):
        super().__init__(config)
        # Hidden layer size
        self.hidden_size = config.hidden_size
        # Natural language bert
        nmodel = code_bert
        # Programming language bert
        # pmodel = code_bert
        # Natural language tokenizer model
        self.ntokenizer = AutoTokenizer.from_pretrained(nmodel)
        self.nbert = AutoModel.from_pretrained(nmodel)
        # Programming language tokenizer model
        self.ptokenizer = self.ntokenizer
        self.pbert =  self.nbert
        
        # Average pooling layer
        self.npool = torch.nn.AdaptiveAvgPool2d((1, self.hidden_size))
        self.ppool = torch.nn.AdaptiveAvgPool2d((1, self.hidden_size))
        
        # Processing concatenated linear layer
        self.dense = torch.nn.Linear(self.hidden_size*3,self.hidden_size)
        # Randomly zero
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # softmax
#         self.softmax = torch.nn.Softmax(dim=1)
        # Classification linear layer
        self.output = torch.nn.Linear(self.hidden_size,2)
        
    
    def forward(self,n_ids,n_attention_mask,p_ids,p_attention_mask):
        n_hidden = self.nbert(n_ids,n_attention_mask)[0]
        p_hidden = self.pbert(p_ids,p_attention_mask)[0]
        # Average pooling
        n_pooler = self.npool(n_hidden).view(-1,self.hidden_size)
        p_pooler = self.ppool(p_hidden).view(-1,self.hidden_size)
        # Concatenation
        gap_hidden = torch.abs(p_pooler-n_pooler)
        concated_hidden = torch.cat((n_pooler,p_pooler),1)
        concated_hidden = torch.cat((concated_hidden,gap_hidden),1)
        # Classification
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
        # Processing concatenated linear layer
        self.dense = torch.nn.Linear(self.hidden_size,self.hidden_size)
        # Randomly zero
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        # Classification linear layer
        self.output = torch.nn.Linear(self.hidden_size,2)
    def forward(self,ids,attention_mask,type_ids):
        # Input to bert
        hidden = self.bert(input_ids=ids,attention_mask=attention_mask,token_type_ids=type_ids)[0]
        # Average pooling
        pooler = self.pool(hidden).view((-1,self.hidden_size))
        # Classification
        # x= self.dropout(pooler)
        # print(hidden.shape, pooler.shape)
        x = self.dense(pooler)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.output(x)
#         x = self.sigmoid(x)
        return x





# # Dataset


# Create dataset class
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
        # Tokenize text data
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


# # Training parameters


class Args:
    def __init__(self,seed,enhance_llm) -> None:
        self.batch_size = 8
        self.lr = 1e-5
        self.epochs = 20 #########################TODO Change to 20
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.momentum = 0.9
        self.gamma = 0.1
        self.k = 10 # K-fold cross validation
        self.patience = 5 # Patience for early stopping
        self.min_delta = 0 # Min delta for early stopping
        # Random seed
        self.seed = seed
        # Source data folder
        self.raw_data_path = '/gemini/data-1'
        # Enhanced data folder
        self.enhance_data_path = f'/gemini/data-2'
        # Encoder model name or path base
        # self.encoder_base = './models/codebert_base'
        # Encoder tokenizer name or path
        # self.tokenizer_name = './models/codebert_base'
        # Model save path, model prediction save path
        self.output_path = f'/gemini/output'
                

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        # Model name_encoder_dataset_enhanced dataset_model for data enhancement
        # Training function parameters should include model name_encoder_dataset name_dataset enhancement method_model for data enhancement



# # Training function



def train_single(df_train,df_test,data_name,model_name,encoder_path,encoder_name,args,split=[0.8,0.1,0.1],enhance_type="none",enhance_llm="none",data_enhance=None,criterion = nn.CrossEntropyLoss()):
    """
    df : Undivided dataset (required)
    data_name : Dataset name (required)
    args : Training parameters
    split : Proportion of dataset division
    enhance_type : Data enhancement method including none req code req(example) code(example)
    enhance_llm : Model name for data enhancement (required)
    data_enhance : Enhanced dataset
    model_name : Model name (required)
    encoder_path : Encoder model path (required)
    encoder_name : Encoder name (required)
    criterion : Loss function
    """
    assert enhance_type in ["none","req","code","req(example)", "code(example)"]
    assert encoder_name in ["CodeBERT","Jina","Jina1024"]
    assert sum(split) == 1

    if enhance_type!="none" and enhance_llm=="none":
        print("Using data enhancement but the model for data enhancement is empty, please re-enter!")
        assert enhance_llm!="none"
    print(f"-----------------{model_name}-{encoder_name}-{data_name}-{enhance_type}-{enhance_llm}--------------------")
    model = Single(BertConfig(),code_bert=encoder_path)

    if encoder_name == "CodeBERT":
        max_length = 512
    if encoder_name == "Jina":
        max_length = 512
    if encoder_name == "Jina1024":
        max_length = 1024
        # model.load_state_dict(torch.load(f'{encoder_path}/Single-Jina.pth'))
    softmax = torch.nn.Softmax()
    
    model = Single(BertConfig(),code_bert=encoder_path)
    if torch.cuda.device_count()>1:
        model = nn.DataParallel(model)
    model.to(args.device)
    

    result_record = pd.read_excel("/gemini/output/result_record.xlsx")
    
    best_val_acc = 0
    best_loss = 100
    # Split dataset
    df_1 = df_train[df_train['label'] == 1]
    df_0 = df_train[df_train['label'] == 0]
    num_1 = len(df_1)
    num_0 = len(df_0)
    train_data = pd.concat([df_1.iloc[:int(split[0]*num_1)],df_0.iloc[:int(split[0]*num_0)]])
    valid_data = pd.concat([df_1.iloc[int(split[0]*num_1):],df_0.iloc[int(split[0]*num_0):]])
    # test_data = pd.read_csv('/gemini/data-3/full_link.csv')
    test_data = df_test
    # test_data = pd.concat([df_1.iloc[int((split[0]+split[1])*num_1):],df_0.iloc[int((split[0]+split[1])*num_0):]])
#     train_data = pd.concat([df[df['label'] == 1].iloc[label_1[0]],df[df['label'] == 0].iloc[label_0[0]]])#.sample(frac=1)
    if enhance_type!="none":
        enhance = data_enhance[['source_text','target_text','label']]#.sample(frac=1)
        # print(enhance)
        train_data = pd.concat([train_data,enhance]).dropna().sample(frac=1)
    print(f"train_num:{len(train_data)},valid_num:{len(valid_data)},test_num:{len(test_data)}")
    tokenizer=AutoTokenizer.from_pretrained(encoder_path)
    train_dataset = MyDataset(train_data,tokenizer,flag='merge',max_length=max_length)
    valid_dataset = MyDataset(valid_data,tokenizer,flag='merge',max_length=max_length)
    test_dataset = MyDataset(test_data,tokenizer,flag='merge',max_length=max_length)
    # Data loader
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,pin_memory=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=True,pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,pin_memory=True)



    # Loss function, optimizer

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    # Early stopping
    # earlyStop = EarlyStopping(args.patience,args.min_delta)
    for epoch in range(args.epochs):
        # ------------------train----------------------------------
        model.train()
        acc = 0
        nums =0
        train_loss_list = []
        for batch in train_data_loader:
            n_ids = batch['ids'].to(args.device)
            n_attention_mask = batch['attention_mask'].to(args.device)

            token_ids = batch['type_ids'].to(args.device)
            label = batch['label'].to(args.device)

            outputs = model(n_ids,n_attention_mask,token_ids)

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

    # End of epoch
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
  
    # Calculate metrics
    result = {}
    # result["model_name"] = model_name
    # result["encoder_name"] = encoder_name
    # result["data_name"] = data_name
    # result["enhance_type"] = enhance_type
    # result["enhance_llm"] = enhance_llm
    # result['seed'] = args.seed
    # result['Accuracy'] = accuracy_score(truth_result,pred_result)
    # result['Precision'] = precision_score(truth_result,pred_result)
    # result['Recall'] = recall_score(truth_result,pred_result)
    # result['F1'] = f1_score(truth_result,pred_result)
    # print(json.dumps(result, indent=4, ensure_ascii=False))

    # Save model
    # torch.save(model.state_dict(), f'{data_name}-{model_name}-{fold}-{f1_score(truth_result,pred_result)}.pth')
    # Delete model
    os.remove(f'{args.output_path}{model_name}-{encoder_name}-{data_name}-{enhance_type}-{enhance_llm}-{args.seed}.pth')
    del list_for_map,pred_result,truth_result,train_data,valid_data,test_data,train_dataset,valid_dataset,test_dataset,train_data_loader,valid_data_loader,test_data_loader

    for _ in range(10):
        torch.cuda.empty_cache()
        
    time.sleep(5)
    del model
    pd.DataFrame(record).to_csv(args.output_path+f'{model_name}-{encoder_name}-{data_name}-{enhance_type}-{enhance_llm}-{args.seed}-record.csv')
    # result_record = result_record.append(result,ignore_index=True)
    # result_record = pd.concat([result_record,pd.DataFrame(result,index=[0])],ignore_index=True)
    # result_record.to_excel("/gemini/output/result_record.xlsx",index=False)
    return record



result = {
    "model_name":[],
    "data_name":[],
    "enhance_type":[],
    "enhance_llm":[],
    "seed":[],
    "req":[],
    "code":[],
    "label":[],
    "label2":[],
    "socre":[]
}

for s in range(2014,2019):
    args = Args(seed=s,enhance_llm="gpt4o") ###
    print(vars(args))
    set_seed(args.seed)
    print(args.device,args.seed,args.enhance_data_path)
    # Training
    for dn in ["EBT","RETRO","eTOUR","iTrust"]:
        # Test set is dngemini/data-3/no_balance_full_EBT_link.csv
        df_test = pd.read_csv(f'/gemini/data-3/no_balance_full_{dn}_link.csv')
        # Training set is the other three datasets
        df_train = pd.concat([pd.read_csv(f'{args.raw_data_path}/full_{dn_}_link.csv') for dn_ in ["EBT","RETRO","eTOUR","iTrust"] if dn_ != dn])
        # Shuffle
        df_train = df_train.sample(frac=1,random_state=s)
        data_enhance=pd.concat([pd.read_excel(f"{args.enhance_data_path}/full_pure_gen_requirement_{dn_}.xlsx") for dn_ in ["EBT","RETRO","eTOUR","iTrust"] if dn_ != dn])
        # df = pd.read_csv(f'{args.raw_data_path}/full_{dn}_link.csv')
        model_name = "Single"
        encoder_name = "CodeBERT"
        encoder_path = "/gemini/pretrain2"
        enhance_llm="GPT-4o" ###
        # Original data
        # train_single(
        #     df_train=df_train,
        #     df_test=df_test,
        #     data_name=dn,
        #     model_name=model_name,
        #     encoder_name=encoder_name,
        #     encoder_path=encoder_path,
        #     args=args
        # )
        
        # No sample - requirement
        record = train_single(
            df_train=df_train,
            df_test=df_test,
            data_name=dn,
            model_name=model_name,
            encoder_name=encoder_name,
            encoder_path=encoder_path,
            args=args,
            enhance_type="req",
            enhance_llm=enhance_llm,
            data_enhance=data_enhance
        )
        result["model_name"].extend([model_name]*len(df_test))
        result["data_name"].extend([dn]*len(df_test))
        result["enhance_type"].extend(['req']*len(df_test))
        result["enhance_llm"].extend([enhance_llm]*len(df_test))
        result["seed"].extend([s]*len(df_test))
        result["req"].extend(df_test['source_text'].tolist())
        result["code"].extend(df_test['target_text'].tolist())
        result["label2"].extend(df_test['label'].tolist())
        result["label"].extend(record["pred"])
        result["socre"].extend(record["score"])
result = pd.DataFrame(result).to_excel("/gemini/output/lopo_codebert_result_record.xlsx",index=False)

        # # No sample - code
        # train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="code",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_gen_Code_{dn}.xlsx"))
        # # Sample - requirement
        # train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="req(example)",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_example_gen_requirement_{dn}.xlsx"))
        # # Sample - code
        # train_single(df=df,data_name=dn,model_name=model_name,encoder_name=encoder_name,encoder_path=encoder_path,args=args,enhance_type="code(example)",enhance_llm=enhance_llm,data_enhance=pd.read_excel(f"{args.enhance_data_path}/full_pure_example_gen_Code_{dn}.xlsx"))
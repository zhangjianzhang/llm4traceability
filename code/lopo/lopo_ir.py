import pandas as pd
import gensim
import random
import numpy as np
from gensim import corpora, models, matutils
from gensim.models import TfidfModel
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from transformers import AutoTokenizer,AutoModel
import torch
import torch.nn as nn
import torch.nn.functional as F
class VSM:
    def __init__(self):
        self.tfidf_model: TfidfModel = None

    def build_model(self, docs_tokens):
        print("Building VSM model...")
        dictionary = corpora.Dictionary(docs_tokens)
        corpus = [dictionary.doc2bow(x) for x in docs_tokens]
        self.tfidf_model = models.TfidfModel(corpus, id2word=dictionary)
        print("Finish building VSM model")

    def _get_doc_similarity(self, doc1_tk, doc2_tk):
        doc1_vec = self.tfidf_model[self.tfidf_model.id2word.doc2bow(doc1_tk)]
        doc2_vec = self.tfidf_model[self.tfidf_model.id2word.doc2bow(doc2_tk)]
        return matutils.cossim(doc1_vec, doc2_vec)

    def get_link_scores(self, source, target):
#         s_tokens = source['tokens'].split()
#         t_tokens = target['tokens'].split()
        score = self._get_doc_similarity(source, target)
        return score
class VSM_CodeBERT:
    def __init__(self,model="myTraceBERT/models/codebert_base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model)
        self.max_length = 512
        self.npool = torch.nn.AdaptiveAvgPool2d((1, 768))
        self.ppool = torch.nn.AdaptiveAvgPool2d((1, 768))
    def get_link_scores(self, source, target):
        nl = self.tokenizer(source, padding='max_length', truncation=True, max_length=self.max_length,return_tensors='pt')
        pl = self.tokenizer(target, padding='max_length', truncation=True, max_length=self.max_length,return_tensors='pt')
        
        nl_ids = nl['input_ids']
        nl_attention_mask = nl['attention_mask']
        pl_ids = pl['input_ids']
        pl_attention_mask = pl['attention_mask']
        
        n_hidden = self.model(nl_ids,nl_attention_mask)[0]
        p_hidden = self.model(pl_ids,pl_attention_mask)[0]
        
        n_pooler = self.npool(n_hidden).view(-1,768).squeeze().detach().numpy()
        p_pooler = self.ppool(p_hidden).view(-1,768).squeeze().detach().numpy()
        
        
        score = np.dot(n_pooler, p_pooler) / (np.linalg.norm(n_pooler)*np.linalg.norm(p_pooler))
        return score

class LDA:
    def __init__(self):
        self.ldamodel = None

    def build_model(self, docs_tokens, num_topics=8,):# passes=1000):
        dictionary = corpora.Dictionary(docs_tokens)
        corpus = [dictionary.doc2bow(x) for x in docs_tokens]
        self.ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary,)
#                                                         passes=passes, alpha='auto',
#                                                         random_state=numpy.random.RandomState(1))

    def get_topic_distrb(self, doc):
        bow_doc = self.ldamodel.id2word.doc2bow(doc)
        return self.ldamodel.get_document_topics(bow_doc)

    def get_link_scores(self, source, target):
        """
        :param doc1_tk: Preprocessed documents as tokens
        :param doc2_tk: Preprocessed documents as tokens
        :return:
        """
#         doc1_tk = source['tokens'].split()
#         doc2_tk = target['tokens'].split()
        dis1 = self.get_topic_distrb(source)
        dis2 = self.get_topic_distrb(target)
        # return 1 - matutils.hellinger(dis1, dis2)
        return matutils.cossim(dis1, dis2)


class LSI:
    def __init__(self):
        self.lsi = None

    def build_model(self, docs_tokens, num_topics=8):
        dictionary = corpora.Dictionary(docs_tokens)
        corpus = [dictionary.doc2bow(x) for x in docs_tokens]
        self.lsi = gensim.models.LsiModel(corpus, num_topics=num_topics, id2word=dictionary)

    def get_topic_distrb(self, doc):
        bow_doc = self.lsi.id2word.doc2bow(doc)
        return self.lsi[bow_doc]

    def get_link_scores(self,  source, target):
#         doc1_tk = source['tokens'].split()
#         doc2_tk = target['tokens'].split()
        dis1 = self.get_topic_distrb(source)
        dis2 = self.get_topic_distrb(target)
        return matutils.cossim(dis1, dis2)

    



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False
set_seed(2014)


import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Download stop words list
nltk.download('stopwords')
nltk.download('punkt_tab')  # Add this line


# Define stop words
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Replace non-English characters and non-numeric characters with spaces
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = text.replace("  "," ")
    text = text.lower()
    
    
    # Split text into words
    words = word_tokenize(text)
    
    # Remove stop words
    filtered_words = [word for word in words if word not in stop_words]
   
    
    return filtered_words

# Sample text
sample_text = "Hello! This is a sample text, including numbers like 123 and symbols like #@$."

# Preprocess text
processed_text = preprocess_text(sample_text)
print("Original text:")
print(sample_text)
print("\nPreprocessed text:")
print(processed_text)



result = {
    "model_name":[],
    "data_name":[],
    "enhance_type":[],
    "enhance_llm":[],
    "req":[],
    "code":[],
    "label":[],
    "socre":[]
}
type2df = {
    "req":"full_pure_gen_requirement_",
    "code":"full_pure_gen_Code_",
    "req(example)":"full_pure_example_gen_requirement_",
    "code(example)":"full_pure_example_gen_Code_"
}
for llm in ["gpt4o"]:#["claude3","gemini","gpt3.5","gpt4o"]:
    for d in ["EBT","eTOUR","iTrust","RETRO"]:
        for t in ["req"]:# ["req","code","req(example)","code(example)"]:
            print(d)
            df= pd.concat([pd.read_csv(f'/home/bigchill/trace_rag/datasets/raw/full_{dn_}_link.csv') for dn_ in ["EBT","RETRO","eTOUR","iTrust"] if dn_ != d])

            enhance_df = pd.concat([pd.read_excel(f"/home/bigchill/trace_rag/datasets/{llm}/{type2df[t]}{d}.xlsx") for dn_ in ["EBT","RETRO","eTOUR","iTrust"] if dn_ != d])
            df2 = pd.concat([df,enhance_df]).reset_index()
            df["source_tokens"] = df["source_text"].apply(preprocess_text)
            df["target_tokens"] = df["target_text"].apply(preprocess_text)
            df2["source_tokens"] = df2["source_text"].apply(preprocess_text)
            df2["target_tokens"] = df2["target_text"].apply(preprocess_text)
            vocab = []
            for i in range(len(df)):
                vocab.append(df["source_tokens"].iloc[i])
                vocab.append(df["target_tokens"].iloc[i])
            vsm = LSI()
            vsm.build_model(vocab)
            df["score"] = df[["source_tokens","target_tokens"]].apply(lambda row:vsm.get_link_scores(row["source_tokens"],row["target_tokens"]),axis=1)
            df2["score"] = df2[["source_tokens","target_tokens"]].apply(lambda row:vsm.get_link_scores(row["source_tokens"],row["target_tokens"]),axis=1)
            df_1 = df[df['label'] == 1]
            df_0 = df[df['label'] == 0]
            num_1 = len(df_1)
            num_0 = len(df_0)
        #     threshold = df_1.iloc[:int(split[0]*num_1)]["score"].values.tolist()
            # train_data = pd.concat([df_1.iloc[:int(split[0]*num_1)],df_0.iloc[:int(split[0]*num_0)]])
            # valid_data = pd.concat([df_1.iloc[int(split[0]*num_1):int((split[0]+split[1])*num_1)],df_0.iloc[int(split[0]*num_0):int((split[0]+split[1])*num_0)]])
            # test_data = pd.concat([df_1.iloc[int(0.9*num_1):],df_0.iloc[int(0.9*num_0):]])
            test_data = pd.read_csv(f'/home/bigchill/trace_rag/datasets/raw/no_balance_full_{d}_link.csv')
            test_data['source_tokens'] = test_data['source_text'].apply(preprocess_text)
            test_data['target_tokens'] = test_data['target_text'].apply(preprocess_text)
            test_data['score'] = test_data[["source_tokens","target_tokens"]].apply(lambda row:vsm.get_link_scores(row["source_tokens"],row["target_tokens"]),axis=1)
            test_data = test_data.reset_index()
            record = {"label":[],"pred":[],"score":[]}
        #     for th in [i/100.0 for i in list(range(0,100))]:
            right = 0
            labels = []
            preds = []
            scores =[]
            for n in range(len(test_data)):
                result["model_name"].append('LSI')
                result["data_name"].append(d)
                result["enhance_type"].append(t)
                result["enhance_llm"].append(llm)
                result["req"].append(test_data["source_text"][n])
                result["code"].append(test_data["target_text"][n])
                result["label"].append(int(test_data["label"][n]))
                result["socre"].append(test_data["score"][n])
pd.DataFrame(result).to_excel("lopo_LSI.xlsx",index=False)
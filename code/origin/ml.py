import random
import numpy as np
import pandas as pd
import gensim
from gensim import corpora, models, matutils
from gensim.models import TfidfModel
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

set_seed(2014)


import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# Download stopwords list
# nltk.download('stopwords')

# Define stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Replace non-English and non-numeric characters with spaces
    text = re.sub(r'[^a-zA-Z0-9]', ' ', text)
    text = text.replace("  "," ")
    text = text.lower()
    
    
    # Split text into words
    words = word_tokenize(text)
    
    # Remove stopwords
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


class ML:
    def __init__(self,model_name,data_name,seed):
        self.tfidf_model: TfidfModel = None
        self.model_name = model_name
        self.data_name = data_name
        self.vec = None
        self.seed = seed
        if model_name =="KNN":
            from sklearn.neighbors import KNeighborsClassifier
            self.ml = KNeighborsClassifier(n_neighbors=5)
        if model_name == "LR":
            from sklearn.linear_model import LogisticRegression
            self.ml = LogisticRegression(random_state=self.seed)
        if model_name == "SVM":
            from sklearn.svm import SVC
            self.ml = SVC(random_state=self.seed)
            
            

    def train(self, df,enhance_type,enhance_llm,data_enhance=None):
        df_1 = df[df['label'] == 1]
        df_0 = df[df['label'] == 0]
        num_1 = len(df_1)
        num_0 = len(df_0)
        train_data = pd.concat([df_1.iloc[:int(0.8*num_1)],df_0.iloc[:int(0.8*num_0)]])
        valid_data = pd.concat([df_1.iloc[int(0.8*num_1):int((0.9)*num_1)],df_0.iloc[int(0.8*num_0):int((0.9)*num_0)]])
        test_data = pd.concat([df_1.iloc[int(0.9*num_1):],df_0.iloc[int(0.9*num_0):]])
        if enhance_type!="none":
            enhance = data_enhance[['source_text','target_text','label']]#.sample(frac=1)
            # print(enhance)
            train_data = pd.concat([train_data,enhance]).dropna().sample(frac=1)
        train_num = len(train_data)
        valid_num = len(valid_data)
        test_num = len(test_data)
        source_target_tokens = train_data["source_target_tokens"].values.tolist() + valid_data["source_target_tokens"].values.tolist() + test_data["source_target_tokens"].values.tolist()
        labels = train_data["label"].values.tolist() + valid_data["label"].values.tolist() + test_data["label"].values.tolist()
        
#         print(f"Building {self.model_name} model...")
        dictionary = corpora.Dictionary(source_target_tokens)
        corpus = [dictionary.doc2bow(x) for x in source_target_tokens]
        self.tfidf_model = models.TfidfModel(corpus, id2word=dictionary)
        vecs = [self.tfidf_model[d] for d in corpus]
#         print(self.vec)
        num_terms = len(dictionary)
        self.vec = [self.vec2dense(vec,num_terms) for vec in vecs]
 
        
        x = np.array(self.vec)
        y = np.array(labels)
        self.ml.fit(x[:train_num],y[:train_num])
        pred = self.ml.predict(x[train_num+valid_num:])
        label_pred = y[train_num+valid_num:]
        acc = accuracy_score(label_pred,pred)
        pre = precision_score(label_pred,pred)
        rec = recall_score(label_pred,pred)
        f1 = f1_score(label_pred,pred)
        return {
            "model_name":self.model_name,
            "data_name":self.data_name,
            "enhance_type":enhance_type,
            "enhance_llm":enhance_llm,
            "seed":self.seed,
            "Accuracy":acc,
            "Precision":pre,
            "Recall":rec,
            "F1":f1
        }
        
    # Convert TF-IDF vector to dense matrix format
    def vec2dense(self,vec, num_terms):
        dense = [0] * num_terms
        for idx, value in vec:
            dense[idx] = value
        return dense
    



result = []
type2df = {
    "req":"full_pure_gen_requirement_",
    "code":"full_pure_gen_Code_",
    "req(example)":"full_pure_example_gen_requirement_",
    "code(example)":"full_pure_example_gen_Code_"
}
for seed in range(2014,2019):
    set_seed(seed)
    for llm in ["claude3","gemini","gpt3.5","gpt4o"]:
        for d in ["EBT","eTOUR","iTrust","RETRO"]:
            for t in ["req","code","req(example)","code(example)"]:
                for model_name in ["LR","SVM","KNN"]: 
                    df = pd.read_csv(f'myTraceBERT/datasets/raw/full_{d}_link.csv')
                    df["source_tokens"] = df["source_text"].apply(preprocess_text)
                    df["target_tokens"] = df["target_text"].apply(preprocess_text)
                    df["source_target_tokens"] = df["source_tokens"]+df["target_tokens"]
                    enhance_df = pd.read_excel(f"myTraceBERT/datasets/{llm}/{type2df[t]}{d}.xlsx")
                    ml = ML(model_name,d,seed)
                    result.append(ml.train(df,enhance_type=t,enhance_llm=llm,data_enhance=enhance_df))
                    print(result[-1])


pd.DataFrame(result).groupby(by=["model_name","data_name","enhance_type","enhance_llm"]).mean()


pd.DataFrame(result).to_excel("myTraceBERT/outputs/ml_enhance_result_record.xlsx",index=False)



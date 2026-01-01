import os

os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'
from typing import List, Dict
import pandas as pd
# LangChain related components
from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings  # Add OpenAI Embeddings import
import re

import os
from openai import OpenAI

# First set environment variable (can be set in system environment variables)
os.environ['OPENAI_API_KEY'] = 'XXXXXXXXXXXXX'

client = OpenAI(
    api_key=os.environ.get('OPENAI_API_KEY'),

    base_url="XXXXXXXXXXXXXXXxx",
    timeout=300
)

def cot(source, target):
    response = client.chat.completions.create(
        model="gpt-4o",
        stream=False,
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": 
             f"""Below are two artifacts from the same software system. Is there a traceability link between (1) and (2)? Give your reasoning and then answer with 'yes' or 'no' enclosed in <trace> </trace>.
(1) Requirement: '''{source}'''
(2) Code: '''{target}'''"""}
        ]
    )
    return response.choices[0].message.content

class LocalCodeSearcher:
    def __init__(self, model_name: str = "text-embedding-3-large"):
        """
        Initialize code searcher
        :param model_name: OpenAI embedding model name (default text-embedding-3-large)
        """
        print(f"Loading embedding model: {model_name} ...")
        # Use OpenAI embeddings instead of HuggingFace embeddings
        self.embeddings = OpenAIEmbeddings(
            model=model_name,
            api_key=os.environ.get('OPENAI_API_KEY'),
            base_url="https://www.vivaapi.cn/v1"  # Add this line, consistent with client
        )
        self.vector_store = None

    def load_and_index_files(self, root_path: str, file_extensions: List[str] = None):
        """
        Load local files and build index
        :param root_path: Code root directory path
        :param file_extensions: List of file extensions to scan, e.g. ['.py', '.js']
        """
        if file_extensions is None:
            file_extensions = ['.py', '.md', '.java', '.cpp', '.js', '.ts']

        print(f"Scanning folder: {root_path} ...")
        documents = []

        # 1. Traverse folder to read files
        for dirpath, _, filenames in os.walk(root_path):
            for file in filenames:
                # Filter file extensions
                if not any(file.endswith(ext) for ext in file_extensions):
                    continue
                
                file_path = os.path.join(dirpath, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        # Record source metadata
                        documents.append(Document(
                            page_content=content,
                            metadata={"source": file_path, "filename": file}
                        ))
                except Exception as e:
                    # Skip files that cannot be read (e.g. binary files or encoding errors)
                    print(f"Skipping file {file}: {e}")

        if not documents:
            print("No matching files found.")
            return

        print(f"Loaded {len(documents)} files, performing code chunking...")

        # 2. Code chunking
        # Use code-optimized chunker, here using Python as an example, general text can also be used
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JAVA, 
            chunk_size=30000000,  # Each chunk about 1000 characters
            chunk_overlap=200 # Overlap 200 characters to maintain context coherence
        )
        
        # If mainly searching other languages, can use general chunker:
        # splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        chunks = splitter.split_documents(documents)
        print(f"Chunking complete, generated {len(chunks)} code snippets. Building vector index...")

        # 3. Build vector database (FAISS)
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        print("Index building complete!\n")

    def search(self, query: str, top_k: int = 3):
        """
        Search code based on text description
        :param query: User input natural language requirement
        :param top_k: Number of results to return
        """
        if not self.vector_store:
            print("Error: Please call load_and_index_files first to build index.")
            return

        # print(f"Retrieving: '{query}' ...")
        # Perform similarity search
        # k=top_k returns k most similar snippets
        results = self.vector_store.similarity_search_with_score(query, k=top_k)

        # print(f"\n{'='*20} Search Results {'='*20}")
 
        cot_answer = {
            "code":[],
            "response":[]
        }
        for idx, (doc, score) in enumerate(results):


            cot_result = cot(query,doc.page_content.strip())
            # if 'yes' in cot_result:
            #     cot_answer.append(f"{doc.metadata.get('filename')}".split('.')[0])

            cot_answer["code"].append(doc.page_content)
            cot_answer["response"].append(cot_result)
        return cot_answer

# Regular expression to match JSON-like structure
def extract_json(text):
    pattern = r'\{\s*"related"\s*:\s*"([^"]+)"\s*,\s*"similarity"\s*:\s*([0-9]*\.?[0-9]+)\s*\}'

    match = re.search(pattern, text)
    if match:
        related = match.group(1)
        similarity = float(match.group(2))
        result = {"related": related, "similarity": similarity}
        return result
    else:
        print("No matching content found")
        return None
# ================= Usage Example =================

if __name__ == "__main__":
    # 1. Initialize searcher
    result = {
    "model_name":[],
    "data_name":[],
    "enhance_type":[],
    "enhance_llm":[],
    "seed":[],
    "req":[],
    "code":[],
    "cot_response":[],
    # "label":[],
    # "label2":[],
    # "socre":[]
}    

    # 2. Set your local code folder path
    # Please replace './my_code_base' with your actual code folder path
    # For demonstration, we assume there's a 'sample_code' folder in current directory
    for cate in ['EBT','eTOUR','iTrust','RETRO']:
        rag = LocalCodeSearcher()
        target_folder = f"/home/bigchill/trace_rag/datasets/raw/code/{cate}" 
        # 3. Build index
        rag.load_and_index_files(target_folder) # TODO
        print(f"{cate} indexing complete")
        test = pd.read_csv(f"/home/bigchill/trace_rag/datasets/raw/no_balance_full_{cate}_link.csv")
        test_1 = test[test['label']==1]
        k = int(test_1.groupby(by='source_text').size().max())
        print(f"{cate} {k}")
        querys = test_1.drop_duplicates(subset='source_text').reset_index(drop=True)
        for q in range(len(querys)):
            query = querys.iloc[q]['source_text']
            # print(query)
            cot_answer = rag.search(query,top_k=k)
            result['model_name'].extend(['RAG']*k)
            result['data_name'].extend([cate]*k)
            result['enhance_type'].extend(['-']*k)
            result['enhance_llm'].extend(['-']*k)
            result['seed'].extend(['-']*k)
            result['req'].extend([query]*k)
            result['cot_response'].extend(cot_answer['response'])
            # print(result)
            pd.DataFrame(result).to_excel(f"lopo_rag_result.xlsx",index=False)
        # test['answer'] = test['source_text'].apply(lambda x: rag.search(x,top_k=4))
        # test.to_csv(f"./all_testset/test_{cate}_link_answer.csv",index=False)
        del rag
    df = pd.read_excel('/home/bigchill/trace_rag/lopo_rag_result.xlsx')

    # text = df['cot_response'][0]
   
    df['cot_json'] = df['cot_response'].apply(extract_json)
    # Find where cot_json is null
    # print(df[df['cot_json'].isnull()]['cot_response'].values[0])
    df['cot_pred'] = df['cot_json'].apply(lambda x: 1 if x['related']=='yes' else 0)
    df['cot_score'] = df['cot_json'].apply(lambda x: x['similarity'])
    df.to_excel('/home/bigchill/trace_rag/lopo_rag_result_1.xlsx',index=False)
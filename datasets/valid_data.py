import sys
import pandas as pd
import os
import json
def normalize_artifact(text: str) -> str:
    if not text:
        return ""
    
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = text.strip()
    
    return text

def filter_synthetic_artifacts(synthetic_links: list, original_pool: set, target_field: str) -> list:
    valid_variants = []
    discarded_count = 0

    for link in synthetic_links:
        generated_artifact = link.get(target_field, "")
        
        norm_artifact = normalize_artifact(generated_artifact)
        
        is_exact_replica = norm_artifact in original_pool
        
        if is_exact_replica:
            discarded_count += 1
            continue 
        else:
            valid_variants.append(link)
            
    return valid_variants, discarded_count

def construct_augmented_dataset(original_data: list, 
                                enhance_data: list, 
                                enhance_type: str) -> list:
   
    original_req_pool = {normalize_artifact(item['source_text']) for item in original_data}
    original_code_pool = {normalize_artifact(item['target_text']) for item in original_data}
    
    # print(">>> Stage 2: Dataset Enrichment Pipeline...")
    if enhance_type == 'code':
    # --- R-to-C Synthetic Pairs ---
        valid_r2c, r2c_discarded = filter_synthetic_artifacts(
            synthetic_links=enhance_data,
            original_pool=original_code_pool,
            target_field='code'
        )
        return valid_r2c
    elif enhance_type == 'req':
    # --- C-to-R Synthetic Pairs ---
        valid_c2r, c2r_discarded = filter_synthetic_artifacts(
            synthetic_links=enhance_data,
            original_pool=original_req_pool,
            target_field='req'
        )
        return valid_c2r
    else:
        raise ValueError("Invalid Enhance Type")
for d in ['EBT','eTOUR','iTrust','RETRO']:
    ori_df = pd.read_csv(f'raw/{d}_link.csv')
    ori_json = ori_df.to_json(orient='records')
    ori_json = json.loads(ori_json) # list
    for llm in ['claude3', 'gemini', 'gpt3.5', 'gpt4o']:
        for t in ['example_gen_Code','example_gen_requirement','gen_Code','gen_requirement']:
            enhance_df = pd.read_excel(f'{llm}/pure_{t}_{d}.xlsx')
            enhance_json = enhance_df.to_json(orient='records')
            enhance_json = json.loads(enhance_json)
            if 'requirement' in t:
                enhance_type = 'req'
            else:
                enhance_type = 'code'
            final_dataset = construct_augmented_dataset(ori_json, enhance_json, enhance_type)
            pd.read_json(json.dumps(final_dataset)).to_csv(f'{llm}/valid_{t}_{d}.csv', index=False)

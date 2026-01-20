import re
from datasets import load_dataset
from typing import Literal 
import os
import pandas as pd

def extract_answer(text):
    match = re.search(r'####\s*(\d+)', text)
    if match:
        return match.group(1)
    else:
        return "No matching answer."
    
def format_example_gsm8k(dataset, idx): 
    question = dataset[idx]["question"]
    answer = extract_answer(dataset[idx]["answer"])
    return question, answer

def gen_gsm8k_dataset(data_dir, phase: Literal["train", "test"]): 
    dataset = []
    splits = {'train': os.path.join(data_dir, "main", 'train-00000-of-00001.parquet'), 
              'test': os.path.join(data_dir, "main", 'test-00000-of-00001.parquet')}

    data_df = pd.read_parquet(splits[phase])
    for i in range(data_df.shape[0]): 
        question, answer = data_df.iloc[i, 0], data_df.iloc[i, 1]
        dataset.append((question, answer))
    return dataset

import re
import ray
import numpy as np
import pandas as pd 
from ray.data import Dataset
from sklearn.model_selection import train_test_split
from src.config import STOPWORDS 
from typing import Dict,List,Tuple
from transformers import BertTokenizer

def data(dataset_loc: str, num_samples: int = None) -> ray.data.Dataset:
    dataset = ray.data.read_csv(dataset_loc)
    dataset = dataset.random_shuffle(seed = 234)
    dataset = ray.data.from_items(dataset.take(num_samples)) if num_samples else dataset
    return dataset

def stratify_split(dataset: ray.data.Dataset, test_size: float, stratify_by: str) -> tuple:
    grouped_ds = dataset.groupby(stratify_by)
    train_ds, val_ds = grouped_ds.split_proportion(1.0 - test_size, seed=42)
    return train_ds, val_ds

def clean_text(text: str) -> str: 
    # change every words into lower case
    text = text.lower()

    # removing stopwords such as "is", "the" and so on
    pattern = re.compile(r'\b(' + r"|".join(STOPWORDS)+ r")\b\s*")
    text = pattern.sub('', text)

    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text) # add space 
    text = re.sub("[^A-Za-z0-9]+", " ", text) # remove other than words and numbers 
    text = re.sub(" +", " ", text) # remove all extra spaces 
    text = text.strip() # remove spaces at the start and at the end 
    text = re.sub(r"http\S+", "", text) # remove links 

    return text

def tokenize(batch: Dict) -> Dict: 
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict= False)
    encoded = tokenizer(batch["text"].tolist(), return_tensors="np", padding = "longest")
    return dict(ids=encoded["input_ids"], masks=encoded["attention_mask"], targets=np.array(batch["label"]))

def preprocess(df: pd.DataFrame):
    df["text"] = df.text.apply(clean_text)
    targets = tokenize(df)
    return targets 
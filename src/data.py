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
    dataset = dataset.random_shuffle(seed = 1234)
    dataset = ray.data.from_items(dataset.take(num_samples)) if num_samples else dataset
    return dataset

def stratify_split(
        ds: Dataset, 
        stratify: str, 
        test_size: float, 
        shuffle: bool = True,
        seed: int = 1234,
) -> Tuple[Dataset, Dataset]: 
    def _add_split(df: pd.DataFrame) -> pd.DataFrame:
        train, test = train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=seed)
        train["_split"]= "train"
        test["_split"] = "test"
        return pd.concat([train,test])
    
    def _filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
        return df[df["_split"] == split].drop("_split", axis = 1)
    
    grouped = ds.groupby(stratify).map_groups(_add_split, batch_format="pandas")
    train_ds = grouped.map_batches(_filter_split, fn_kwargs={"split":"train"}, batch_format="pandas")
    test_ds = grouped.map_batches(_filter_split, fn_kwargs={"split":"test"}, batch_format="pandas")

    train_ds = train_ds.random_shuffle(seed=seed)
    test_ds = test_ds.random_shuffle(seed=seed)

    return train_ds, test_ds

def clean_text(text: str) -> str:
    # change every words into lower case
    text = text.lower()

    text = re.sub(r"http\S+", "", text) # remove links (must run before punctuation gets spaced apart below)

    # removing stopwords such as "is", "the" and so on
    pattern = re.compile(r'\b(' + r"|".join(STOPWORDS)+ r")\b\s*")
    text = pattern.sub('', text)

    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text) # add space
    text = re.sub("[^A-Za-z0-9]+", " ", text) # remove other than words and numbers
    text = re.sub(" +", " ", text) # remove all extra spaces
    text = text.strip() # remove spaces at the start and at the end

    return text

_tokenizer_cache = {}

def _get_tokenizer(model_name: str = "allenai/scibert_scivocab_uncased"):
    # cache per-process so map_batches workers don't reload the tokenizer on every batch
    if model_name not in _tokenizer_cache:
        _tokenizer_cache[model_name] = BertTokenizer.from_pretrained(model_name, return_dict=False)
    return _tokenizer_cache[model_name]

def tokenize(batch: Dict) -> Dict:
    tokenizer = _get_tokenizer()
    encoded = tokenizer(batch["text"].tolist(), return_tensors="np", padding = "longest")
    return dict(ids=encoded["input_ids"], masks=encoded["attention_mask"], targets=np.array(batch["label"]))

def preprocess(df: pd.DataFrame):
    df["text"] = df.text.apply(clean_text)
    targets = tokenize(df)
    return targets 

class CustomPreprocessor():
    """Custom Preprocess."""
    def __init__(self, label_decoder = None):
        self.label_decoder = label_decoder or {
            0: "World",
            1: "Sports",
            2: "Business",
            3: "Sci/Tech"
        }
        self.class_to_index = {v:k for k, v in self.label_decoder.items()}

    def fitting(self, ds):
        _ = ds.unique(column="label")
        return self
    
    def transforming(self, ds):
        return ds.map_batches(
            preprocess, 
            batch_format = "pandas")
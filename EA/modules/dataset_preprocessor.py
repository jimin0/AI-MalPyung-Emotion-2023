import torch

from transformers import AutoTokenizer
from datasets import Dataset

import re
import emoji
from soynlp.normalizer import repeat_normalize

special_tokens = [
    '&others&'
]

pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')


def clean_sentence(x):
    x = pattern.sub(' ', x)
    x = emoji.replace_emoji(x, replace='')  # emoji 삭제
    x = url_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x

def remove_special_tokens(sentence):
    for token in special_tokens:
        sentence = sentence.replace(token, '')
    return sentence

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, encoded_dataset):
        self.encoded_dataset = encoded_dataset

    def __len__(self):
        return len(self.encoded_dataset)

    def __getitem__(self, idx):
        item = self.encoded_dataset[idx]
        input_ids = torch.tensor(item["input_ids"])
        attention_mask = torch.tensor(item["attention_mask"])
        labels = torch.tensor(item["labels"])
        return input_ids, attention_mask, labels


def preprocess_data(tokenizer, max_seq_len, examples,clean):
    labels = [key for key in examples["output"].keys()]
    label2id = {label:idx for idx, label in enumerate(labels)}
    
    # take a batch of texts
    text1 = examples["input"]["form"]
    text2 = examples["input"]["target"]["form"]
    
    if clean is True:
        text1 = clean_sentence(text1)
        text2 = clean_sentence(text2)
    # encode them
    encoding = tokenizer(text1, text2, padding="max_length", truncation=True, max_length=max_seq_len)
    # add labels
    encoding["labels"] = [0.0] * len(labels)
    for key, idx in label2id.items():
        if examples["output"][key] == 'True':
            encoding["labels"][idx] = 1.0
    return encoding

def preprocess_data_special_token(tokenizer, max_seq_len, examples,clean,prompt):
    labels = [key for key in examples["output"].keys()]
    label2id = {label:idx for idx, label in enumerate(labels)}
    
    # take a batch of texts
    text1 = examples["input"]["form"]
    text2 = examples["input"]["target"]["form"]
    
    begin = examples["input"]["target"]["begin"]
    end = examples["input"]["target"]["end"]
    
    if prompt:
        if text2 is not None:
            sentence = f'다음에 올 문장에서 {text1[begin:end]}에 대한 감정은?' + tokenizer.sep_token  + text1[:begin] + '<e>' + text1[begin:end] + '</e>' + text1[end:] 
        else:
            sentence = '다음에 올 문장에서 나타나는 감정은?'  + tokenizer.sep_token + text1
    else: 
        if text2 is not None:
            sentence = text1[:begin] + '<e>' + text1[begin:end] + '</e>' + text1[end:]
        else:
            sentence = text1
    
    sentence = remove_special_tokens(sentence)
    if clean is True:
        sentence = clean_sentence(sentence)

    # encode them
    encoding = tokenizer(sentence, padding="max_length", truncation=True, max_length=max_seq_len)
    # add labels
    encoding["labels"] = [0.0] * len(labels)
    for key, idx in label2id.items():
        if examples["output"][key] == 'True':
            encoding["labels"][idx] = 1.0
    return encoding

def load_and_preprocess_datasets(tokenizer_name, train_path, val_path, max_seq_len: int, special_token, clean):
    ...

    tokenizer = tokenizer_name

    train_ds = Dataset.from_json(train_path)
    if special_token is True:
        encoded_tds = train_ds.map(lambda x: preprocess_data_special_token(tokenizer, max_seq_len, x, clean), remove_columns=train_ds.column_names ,  load_from_cache_file=False)
        if val_path:
            valid_ds = Dataset.from_json(val_path)
            encoded_vds = valid_ds.map(lambda x: preprocess_data_special_token(tokenizer, max_seq_len, x, clean), remove_columns=valid_ds.column_names,  load_from_cache_file=False)
        else:
            encoded_vds = None
    else:
        encoded_tds = train_ds.map(lambda x: preprocess_data(tokenizer, max_seq_len, x, clean), remove_columns=train_ds.column_names)

        if val_path:
            valid_ds = Dataset.from_json(val_path)
            encoded_vds = valid_ds.map(lambda x: preprocess_data(tokenizer, max_seq_len, x, clean), remove_columns=valid_ds.column_names)
        else:
            encoded_vds = None
    return encoded_tds, encoded_vds


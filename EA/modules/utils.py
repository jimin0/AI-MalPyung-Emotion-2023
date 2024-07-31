import os
import json
import torch
from transformers import AutoTokenizer
from datasets import Dataset

def save_as_jsonl(original_dataset, indices, fname):
    data_to_save = [original_dataset[i] for i in indices]
    jsonldump(data_to_save, fname)

def jsonlload(fname):
    with open(fname, "r", encoding="utf-8") as f:
        lines = f.read().strip().split("\n")
        j_list = [json.loads(line) for line in lines]
    return j_list

def jsonldump(j_list, fname):
    with open(fname, "w", encoding='utf-8') as f:
        for json_data in j_list:
            f.write(json.dumps(json_data, ensure_ascii=False)+'\n')

def create_output_dir(output_dir: str):
    """
    Creates the output directory if it doesn't exist.

    Args:
    - output_dir (str): Path of the output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    return f'[+] Save output to "{output_dir}"'

def load_tokenizer(tokenizer_path: str):
    """
    Loads the tokenizer from a given path.

    Args:
    - tokenizer_path (str): Path to the tokenizer.

    Returns:
    - tokenizer: Loaded tokenizer.
    """
    return AutoTokenizer.from_pretrained(tokenizer_path)

def load_datasets(train_path: str, valid_path: str):
    """
    Loads training and validation datasets from given paths.

    Args:
    - train_path (str): Path to the training dataset.
    - valid_path (str): Path to the validation dataset.

    Returns:
    - train_ds, valid_ds: Loaded training and validation datasets.
    """
    train_ds = Dataset.from_json(train_path)
    valid_ds = Dataset.from_json(valid_path)
    return train_ds, valid_ds

def get_labels_from_dataset(dataset):
    """
    Retrieves labels from a given dataset.

    Args:
    - dataset: Input dataset.

    Returns:
    - id2label, label2id: Mappings from id to label and label to id.
    """
    labels = list(dataset["output"][0].keys())
    id2label = {idx: label for idx, label in enumerate(labels)}
    label2id = {label: idx for idx, label in enumerate(labels)}
    return labels, id2label, label2id

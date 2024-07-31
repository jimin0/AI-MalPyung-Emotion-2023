import os
import sys
import torch
import wandb

from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from datasets import Dataset

# module import
from modules.logger_module import get_logger, log_args, setup_seed
from modules.utils import create_output_dir, load_tokenizer, load_datasets, get_labels_from_dataset
from modules.arg_parser import get_args

from modules.utils import jsonldump, jsonlload
from tqdm import tqdm

import numpy as np

# model import
from models.EnhancedPoolingModel2 import EnhancedPoolingModel2
from models.RealAttention5 import RealAttention5

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_classes = {
    'EnhancedPoolingModel2': EnhancedPoolingModel2,
    'RealAttention5': RealAttention5
   
}

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
        return input_ids, attention_mask

def preprocess_data(tokenizer, max_seq_len, examples):

    # take a batch of texts
    text1 = examples["input"]["form"]
    text2 = examples["input"]["target"]["form"]
    
    
    # encode them
    encoding = tokenizer(text1, text2, padding="max_length", truncation=True, max_length=max_seq_len)
    # add labels

    return encoding

def preprocess_data_special_token(tokenizer, max_seq_len, examples, prompt, clean):

    text1 = examples["input"]["form"]
    text2 = examples["input"]["target"]["form"]
    
    begin = examples["input"]["target"]["begin"]
    end = examples["input"]["target"]["end"]
    
    if prompt is True:
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
    
    #clean 예외
    if clean is True:
        sentence = clean_sentence(sentence)
        
    # encode them
    encoding = tokenizer(sentence, padding="max_length", truncation=True, max_length=max_seq_len)
    # add labels
    return encoding

def load_and_preprocess_datasets(tokenizer_name, train_path, max_seq_len: int, special_token, prompt, clean):
    ...

    tokenizer = tokenizer_name

    train_ds = Dataset.from_json(train_path)
    if special_token is True:
        encoded_tds = train_ds.map(lambda x: preprocess_data_special_token(tokenizer, max_seq_len, x, prompt, clean), remove_columns=train_ds.column_names, load_from_cache_file=False)
    else:
        encoded_tds = train_ds.map(lambda x: preprocess_data(tokenizer, max_seq_len, x, prompt, clean), remove_columns=train_ds.column_names, load_from_cache_file=False)

    return encoded_tds

def main(args):

    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)  # type: ignore

    tokenizer = load_tokenizer(args.tokenizer)

    # Load datasets and log the action
    train_ds, _ = load_datasets(args.train_path, args.val_path)

    # Get label mappings
    _, id2label, _ = get_labels_from_dataset(train_ds)

    test_ds =  load_and_preprocess_datasets(tokenizer, args.test_path, args.max_seq_len, args.special_token, args.prompt, args.clean)
    test_dataset = CustomDataset(test_ds)
    data_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    sigmoid = torch.nn.Sigmoid()
    ensemble_predictions = []

    for fold in range(1, 6):
        model_class = model_classes[args.model_class]
        model = model_class(args, len(id2label), tokenizer)

        # 각 fold의 체크포인트 로드
        model_ckpt_path = f"{args.model_ckpt_path}fold_{fold}/model.pt"
        model.load_state_dict(torch.load(model_ckpt_path, map_location=device))
        model.to(device)
        model.eval()

        fold_predictions = []

        for batch in tqdm(data_loader):
            input_ids, attention_mask = batch
            inputs = input_ids.to(device)
            attention_masks = attention_mask.to(device)
            outputs = model(inputs, attention_mask=attention_masks)

            probs = sigmoid(outputs[1]).cpu().detach().numpy()
            fold_predictions.extend(probs)  # Use extend here since probs is 2D array for each batch

        # 모델 삭제하여 메모리 해제
        del model
        torch.cuda.empty_cache()

        # Convert fold_predictions list to numpy array and add to ensemble_predictions
        ensemble_predictions.append(np.array(fold_predictions))

    # Convert ensemble_predictions list to numpy array
    ensemble_predictions = np.array(ensemble_predictions)

    # 앙상블
    avg_predictions = np.mean(ensemble_predictions, axis=0)
    
    j_list = jsonlload(args.test_path)
    for idx, logits in enumerate(avg_predictions):
        j_list[idx]["logits"] = logits.tolist()

    jsonldump(j_list, f'{args.output_jsonl}logits.jsonl')


    
if __name__ == "__main__":
    args = get_args()
    # wandb.init(name=args.wandb_run_name, project=args.wandb_project, entity=args.wandb_entity)
    # wandb.config.update(args)
    exit(main(args))

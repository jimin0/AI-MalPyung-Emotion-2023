import os
import sys
import torch
import wandb

from torch.utils.data import DataLoader, ConcatDataset, Subset
from sklearn.model_selection import KFold
from datasets import concatenate_datasets,Dataset
# from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers import (AutoModel, AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    # Trainer,
    EvalPrediction,
    BitsAndBytesConfig,
)
# module import
from modules.logger_module import get_logger, log_args, setup_seed
from modules.utils import create_output_dir, load_tokenizer, load_datasets, get_labels_from_dataset
from modules.arg_parser import get_args

from modules.utils import jsonldump, jsonlload
from tqdm import tqdm

import numpy as np

from peft import PeftModel, PeftConfig
os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


import re
import emoji
from soynlp.normalizer import repeat_normalize
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from peft import LoraConfig, get_peft_model, TaskType

pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

special_tokens = [
    '&others&'
]

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

def preprocess_data_special_token(tokenizer, max_seq_len, examples, prompt):

    text1 = examples["input"]["form"]
    text2 = examples["input"]["target"]["form"]
    
    begin = examples["input"]["target"]["begin"]
    end = examples["input"]["target"]["end"]
    
    if prompt is True:
        if text2 is not None:
            sentence = f'다음에 올 문장에서 {text1[begin:end]}에 대한 감정은?' +  text1[:begin] + '<e>' + text1[begin:end] + '</e>' + text1[end:] 
        else:
            sentence = '다음에 올 문장에서 나타나는 감정은?'  + text1 
    else:
        if text2 is not None:
            sentence = f'다음에 올 문장에서 {text1[begin:end]}에 대한 감정을 기쁨, 기대, 신뢰, 놀라움, 혐오, 공포, 화남, 슬픔에서 고르시오' +  text1[:begin] + '<e>' + text1[begin:end] + '</e>' + text1[end:] 
        else:
            sentence = '다음에 올 문장에서 나타나는 감정을 기쁨, 기대, 신뢰, 놀라움, 혐오, 공포, 화남, 슬픔에서 고르시오'  + text1 
    
    sentence = remove_special_tokens(sentence)

    # encode them
    encoding = tokenizer(sentence, padding="max_length", truncation=True, max_length=max_seq_len)
    # add labels
    return encoding

def load_and_preprocess_datasets(tokenizer_name, train_path, max_seq_len: int, special_token, prompt):
    ...

    tokenizer = tokenizer_name

    train_ds = Dataset.from_json(train_path)
    if special_token is True:
        encoded_tds = train_ds.map(lambda x: preprocess_data_special_token(tokenizer, max_seq_len, x, prompt), remove_columns=train_ds.column_names,  load_from_cache_file=False)
    else:
        encoded_tds = train_ds.map(lambda x: preprocess_data(tokenizer, max_seq_len, x, prompt), remove_columns=train_ds.column_names,  load_from_cache_file=False)

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

    test_dataset =  load_and_preprocess_datasets(tokenizer, args.test_path, args.max_seq_len, args.special_token, args.prompt)
    test_dataset = CustomDataset(test_dataset)
    data_loader = DataLoader(test_dataset, batch_size=args.batch_size)

    config = PeftConfig.from_pretrained(args.model_ckpt_path)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForSequenceClassification.from_pretrained(
            args.model_path,
            problem_type="multi_label_classification",
            num_labels=8,
            quantization_config=bnb_config,
    )
    model = PeftModel.from_pretrained(model, args.model_ckpt_path)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    torch.set_grad_enabled(False)
    # 각 fold의 체크포인트 로드
    sigmoid = torch.nn.Sigmoid()

    predictions = []
    for batch in tqdm(data_loader):
        input_ids, attention_mask = batch
        oup = model(
            input_ids.to(device),
            attention_mask=attention_mask.to(device)
        )
        
        logits = oup.logits
        probs = sigmoid(logits).cpu().detach().numpy()  # 로짓에 시그모이드 함수를 적용하여 확률을 얻습니다.
        predictions.extend(probs)  # 확률을 predictions 리스트에 추가합니다.

    # JSONL 파일에서 데이터 로드
    j_list = jsonlload(args.test_path)

    # 예측된 확률을 기록합니다.
    for idx, probs in enumerate(predictions):
        j_list[idx]["logits"] = probs.tolist()  # Numpy 배열을 Python 리스트로 변환합니다.

    # 결과를 JSONL 파일에 저장합니다.
    jsonldump(j_list, f'{args.output_jsonl}logits.jsonl')


    
if __name__ == "__main__":
    args = get_args()
    # wandb.init(name=args.wandb_run_name, project=args.wandb_project, entity=args.wandb_entity)
    # wandb.config.update(args)
    exit(main(args))

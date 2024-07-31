import argparse
import torch
from datetime import datetime, timezone, timedelta

kst = timezone(timedelta(hours=9))
train_serial = datetime.now(tz=kst).strftime("%Y%m%d_%H%M%S")

def get_args():
    parser = argparse.ArgumentParser(prog="train", description="Train Table to Text with BART")

    g = parser.add_argument_group("Common Parameter")
    g.add_argument("--output_dir", type=str, default=f'./results/{train_serial}/', help="output directory path to save artifacts")
    g.add_argument("--model_path", type=str, default="kykim/electra-kor-base", help="model file path")
    g.add_argument("--model_class", type=str, default="ElectraModel", help="model class")
    g.add_argument("--tokenizer", type=str, default="kykim/electra-kor-base", help="huggingface tokenizer path")
    g.add_argument("--logging_steps", type=int, default="500",help="logging_steps")
    g.add_argument("--kfold", type=int, default=None, help="Number of k-folds for cross validation")
    g.add_argument("--max-seq-len", type=int, default=256, help="max sequence length form's max seq_len = 218")
    g.add_argument("--batch-size", type=int, default=64, help="training batch size")
    g.add_argument("--valid-batch-size", type=int, default=8, help="validation batch size")
    g.add_argument("--accumulate-grad-batches", type=int, default=1, help=" the number of gradient accumulation steps")
    g.add_argument("--epochs", type=int, default=10, help="the number of training epochs")
    g.add_argument("--learning-rate", type=float, default=2e-5, help="max learning rate")
    g.add_argument("--special_token", type=bool, default=False, help="special_token adding")
    g.add_argument("--classifier_hidden_size", type=int, default=768, help="model hiddensize")
    g.add_argument("--num_layers", type=float, default=3, help="num_layer")
    g.add_argument("--lstm_hidden_size", type=int,default=768 ,help="lstm size")
    g.add_argument("--classifier_dropout_prob", type=float, default=0.1, help="dropout rate")
    g.add_argument("--weight-decay", type=float, default=0.01, help="weight decay")
    g.add_argument("--gpus", type=int, default=0, help="the number of gpus")
    g.add_argument("--n_gram", type=int, default=5, help="the number of gpus")
    g.add_argument("--seed", type=int, default=42, help="random seed")
    g.add_argument("--hidden", type=int, default=128, help="random seed")
    g.add_argument("--clean", type=bool, default=False, help="clean data value")
    g.add_argument("--label_smoothing", type=bool, default=False, help="clean data value")
    g.add_argument("--prompt", type=bool, default=False, help="clean data value")
    g.add_argument("--train_path", type=str, default="./data/train.jsonl", help="train_set path")
    g.add_argument("--val_path", type=str, default="./data/dev.jsonl", help="validation_set path")
    g.add_argument("--test_path", type=str, default="./data/test.jsonl", help="test_set path")
    g.add_argument("--test_labeled_path", type=str, default="./data/test_label1.jsonl", help="test_set path")
    g.add_argument("--kfold_data_path", type=str, default="./data/train+dev.jsonl", help="test_set path")
    g.add_argument("--model_ckpt_path", type=str, default="./results/20230821_205025/fold_5_best_model.pt", help="test_set path")
    g.add_argument("--output_jsonl", type=str, default=f'.test.jsonl', help="output directory path to save artifacts")

    g = parser.add_argument_group("Wandb Options")
    g.add_argument("--wandb_run_name", type=str, help="wanDB run name")
    g.add_argument("--wandb_entity", type=str, default='modu_ai' ,help="wanDB entity name")
    g.add_argument("--wandb_project", type=str, default='MODU_EA' ,help="wanDB project name")

    args = parser.parse_args()

    if not args.wandb_run_name:
        args.wandb_run_name = f'{train_serial}_{args.model_class}'
    
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return args

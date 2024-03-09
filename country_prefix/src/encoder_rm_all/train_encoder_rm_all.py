# python country_prefix/src/encoder_rm_all/train_encoder_rm_all.py --data_dir country_prefix/data --model_name distilbert-base-uncased --batch_size 16 --wd 1e-2 --lr 2e-5 --num_epochs 3 --save_dir country_prefix/train_status/encoder_rm_all

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, Trainer
import pandas as pd
import argparse
import os
import torch

import sys
sys.path.append("country_prefix/src/utils")
from score_ds import Score_DS
from compute_metrics import compute_metrics

def get_train_test_ds(data_dir, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    max_length = tokenizer.max_model_input_sizes[model_name]

    train_df = pd.read_csv(os.path.join(data_dir, "train_ds.csv"), keep_default_na=False)

    train_ds = Score_DS(
        tokenizer=tokenizer, 
        max_length=max_length,
        questions=train_df["question"],
        options=train_df["option"],
        scores=train_df["score"],
        country_ids=train_df["country_id"]
    )

    test_df = pd.read_csv(os.path.join(data_dir, "test_ds.csv"), keep_default_na=False)
    test_ds = Score_DS(
        tokenizer=tokenizer, 
        max_length=max_length,
        questions=test_df["question"],
        options=test_df["option"],
        scores=test_df["score"],
        country_ids=test_df["country_id"]
    )

    return train_ds, test_ds

def train(device, data_dir, model_name, config, save_dir):
    train_ds, test_ds = get_train_test_ds(data_dir, model_name)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1).to(device)

    training_args = TrainingArguments(
        output_dir=os.path.join(save_dir),          
        num_train_epochs = config["num_epochs"],     
        per_device_train_batch_size=config["batch_size"],   
        per_device_eval_batch_size=config["batch_size"],   
        weight_decay=config["wd"],               
        learning_rate=config["lr"],
        logging_dir=os.path.join(save_dir),            
        save_total_limit=1,
        load_best_model_at_end=True,     
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch"
    ) 

    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_ds,         
        eval_dataset=test_ds,          
        compute_metrics=compute_metrics    
    )

    print(f"====Model device: {trainer.model.device}====")

    trainer.train()

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    config = {
        "batch_size": args.batch_size,
        "wd": args.wd,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "wd": args.wd
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train(device, args.data_dir, args.model_name, config, args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--wd", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()

    main(args)
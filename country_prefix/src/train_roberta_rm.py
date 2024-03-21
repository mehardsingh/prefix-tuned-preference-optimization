# python country_prefix/src/train_encoder_rm.py --data_dir country_prefix/data --model_name distilbert/distilroberta-base --batch_size 16 --wd 1e-2 --lr 2e-5 --num_epochs 3 --save_dir country_prefix/train_status/encoder_rm

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, AutoConfig
import os
import sys
import torch
import argparse
from peft import LoraConfig, get_peft_model, TaskType
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers.modeling_utils import PreTrainedModel

sys.path.append("train_utils")
from score_ds import get_train_test_ds
from compute_metrics import compute_metrics

class RobertaRM(AutoModelForSequenceClassification):
    def __init__(self, config):
        super().__init__(config)
        self.sigmoid = nn.Sigmoid()

    def forward(self, **kwargs):
        outputs = super().forward(**kwargs)
        logits = outputs.logits
        probabilities = self.sigmoid(logits.squeeze(-1))
        return probabilities

def prepare_roberta(country_prefix, lora, model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if country_prefix == "True":
        join_fn = lambda country, question, option: "[{}] {} {} {} {}".format(country, tokenizer.sep_token, question, tokenizer.sep_token, option)
    else:
        join_fn = lambda country, question, option: "{} {} {}".format(question, tokenizer.sep_token, option)
    
    config = AutoConfig.from_pretrained(model_name, num_labels=1)
    model = RobertaRM.from_pretrained(model_name, config=config)

    if lora == "True":
        lora_config = LoraConfig(
            r=64, 
            lora_alpha=16, 
            target_modules = ['query', 'key', 'value'],
            lora_dropout=0.1, 
            bias="none", 
            task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    return model, tokenizer, join_fn

def train(country_prefix, lora, country, data_dir, model_name, config, save_dir):
    if "roberta" in model_name:
        model, tokenizer, join_fn = prepare_roberta(country_prefix, lora, model_name)
    else:
        raise ValueError(f"No classification for model_name: {model_name}")

    train_ds, test_ds, _, _ = get_train_test_ds(country, data_dir, tokenizer, join_fn, max_length=256)
    
    output_dir = os.path.join(save_dir, country) if country else os.path.join(save_dir, "all")
    training_args = TrainingArguments(
        output_dir=output_dir,       
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

    print(f"Device Used: {training_args.device}")

    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_ds,         
        eval_dataset=test_ds,          
        compute_metrics=compute_metrics    
    )

    trainer.train()
    trainer.save_model(output_dir)

def main(args):
    print(f"Cuda Availability: {torch.cuda.is_available()}")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    config = {
        "batch_size": args.batch_size,
        "wd": args.wd,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "wd": args.wd
    }

    train(args.country_prefix, args.lora, args.country, args.data_dir, args.model_name, config, args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country_prefix", default="False", type=str)
    parser.add_argument("--lora", default="False", type=str)
    parser.add_argument("--country", default=None, type=str)
    parser.add_argument("--data_dir", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--wd", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--num_epochs", type=int)
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()

    main(args)

# python country_prefix/src/train_decoder_rm.py --data_dir country_prefix/data --model_name meta-llama/Llama-2-7b-hf --batch_size 16 --wd 1e-2 --lr 2e-5 --num_epochs 3 --save_dir country_prefix/train_status/decoder_rm_all

from transformers import LlamaForSequenceClassification, TrainingArguments, Trainer, LlamaTokenizerFast, AutoTokenizer, BitsAndBytesConfig
import os
import sys
import torch
import argparse
from peft import LoraConfig, PrefixTuningConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

sys.path.append("train_utils")
from score_ds import get_train_test_ds
from compute_metrics import compute_metrics
from score_head import ScoreHead

def prepare_llama(quantize, lora, model_name):
    tokenizer = LlamaTokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    join_fn = lambda q, r : f"[INST]\nQuestion: {q}[/INST]\nResponse: {r}"

    if quantize:
        model = LlamaForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=1,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
            ),
            torch_dtype=torch.bfloat16,
        )
        model = prepare_model_for_kbit_training(model)

    else:
        model = LlamaForSequenceClassification.from_pretrained(
            model_name, 
            num_labels=1,
            torch_dtype=torch.bfloat16
        )

    if lora:
        lora_config = LoraConfig(
            r=64, 
            lora_alpha=16, 
            target_modules = ['q_proj', 'k_proj', 'down_proj', 'v_proj', 'gate_proj', 'o_proj', 'up_proj'],
            lora_dropout=0.1, 
            bias="none", 
            task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(model, lora_config)

    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    model.score = ScoreHead(model.config.hidden_size)

    return model, tokenizer, join_fn

def train(quantize, lora, country, data_dir, model_name, config, save_dir):
    if "Llama" in model_name:
        model, tokenizer, join_fn = prepare_llama(quantize, lora, model_name)
    else:
        raise ValueError(f"No classification for model_name: {model_name}")

    train_ds, test_ds = get_train_test_ds(country, data_dir, model_name, tokenizer, join_fn, max_length=512)
    
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

    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_ds,         
        eval_dataset=test_ds,          
        compute_metrics=compute_metrics    
    )

    trainer.train()

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        "batch_size": args.batch_size,
        "wd": args.wd,
        "lr": args.lr,
        "num_epochs": args.num_epochs,
        "wd": args.wd
    }

    train(args.quantize, args.lora, args.country, args.data_dir, args.model_name, config, args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantize", default="True", type=str)
    parser.add_argument("--lora", default="True", type=str)
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
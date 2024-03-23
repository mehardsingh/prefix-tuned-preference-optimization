from torch.utils.data import Dataset
import torch
from transformers import AutoTokenizer
import pandas as pd
import os

class Score_DS(Dataset):
    def __init__(self, tokenizer, join_fn, max_length, questions, options, scores, country_ids):
        self.tokenizer = tokenizer
        self.join_fn = join_fn
        self.max_length = max_length

        self.questions = questions
        self.options = options
        self.scores = scores
        self.country_ids = country_ids
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        option = self.options[idx]
        score = self.scores[idx]
        country_id = self.country_ids[idx]

        # prompt = question + self.tokenizer.sep_token + option
        prompt = self.join_fn(question, option)
        prompt_enc = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        
        return {
            'input_ids': prompt_enc['input_ids'].squeeze(),
            'attention_mask': prompt_enc['attention_mask'].squeeze(),
            'labels': torch.tensor(score, dtype=torch.float),
            'country_id': torch.tensor(country_id, dtype=torch.int64)
        }
    
def get_train_test_ds(country, data_dir, model_name, tokenizer, join_fn, max_length):
    train_df = pd.read_csv(os.path.join(data_dir, "train_ds.csv"), keep_default_na=False)
    test_df = pd.read_csv(os.path.join(data_dir, "test_ds.csv"), keep_default_na=False)

    if country:
        train_df = train_df[train_df["country"] == country]
        test_df = test_df[test_df["country"] == country]

    print("Train-Test Split: {:.2f}-{:.2f}%".format(100*len(train_df)/(len(train_df)+len(test_df)), 100*len(test_df)/(len(train_df)+len(test_df))))
    # tokenizer, join_fn, max_length, questions, options, scores, country_ids
    train_ds = Score_DS(
        tokenizer=tokenizer, 
        join_fn=join_fn,
        max_length=max_length,
        questions=list(train_df["question"]),
        options=list(train_df["option"]),
        scores=list(train_df["score"]),
        country_ids=list(train_df["country_id"])
    )

    test_ds = Score_DS(
        tokenizer=tokenizer, 
        join_fn=join_fn,
        max_length=max_length,
        questions=list(test_df["question"]),
        options=list(test_df["option"]),
        scores=list(test_df["score"]),
        country_ids=list(test_df["country_id"])
    )

    return train_ds, test_ds, train_df, test_df
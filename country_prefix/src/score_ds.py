from torch.utils.data import Dataset
import torch
import pandas as pd
import os

class Score_DS(Dataset):
    def __init__(self, tokenizer, join_fn, max_length, questions, options, scores, countries):
        self.tokenizer = tokenizer
        self.join_fn = join_fn
        self.max_length = max_length

        self.questions = questions
        self.options = options
        self.scores = scores
        self.countries = countries
    
    def __len__(self):
        return len(self.questions)
    
    def __getitem__(self, idx):
        question = self.questions[idx]
        option = self.options[idx]
        score = self.scores[idx]
        country = self.countries[idx]

        prompt = self.join_fn(country, question, option)
        prompt_enc = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        
        return {
            'input_ids': prompt_enc['input_ids'].squeeze(),
            'attention_mask': prompt_enc['attention_mask'].squeeze(),
            'labels': torch.tensor(score, dtype=torch.float),
            'countries': country
        }
    
def get_train_test_ds(country, data_dir, tokenizer, join_fn, max_length):
    train_df = pd.read_csv(os.path.join(data_dir, "train_ds.csv"), keep_default_na=False)
    test_df = pd.read_csv(os.path.join(data_dir, "test_ds.csv"), keep_default_na=False)

    if country:
        train_df = train_df[train_df["country"] == country]
        test_df = test_df[test_df["country"] == country]

    print("Train-Test Split: {:.2f}-{:.2f}%".format(100*len(train_df)/(len(train_df)+len(test_df)), 100*len(test_df)/(len(train_df)+len(test_df))))

    train_ds = Score_DS(
        tokenizer=tokenizer, 
        join_fn=join_fn,
        max_length=max_length,
        questions=list(train_df["question"]),
        options=list(train_df["option"]),
        scores=list(train_df["score"]),
        countries=list(train_df["country"])
    )

    test_ds = Score_DS(
        tokenizer=tokenizer, 
        join_fn=join_fn,
        max_length=max_length,
        questions=list(test_df["question"]),
        options=list(test_df["option"]),
        scores=list(test_df["score"]),
        countries=list(test_df["country"])
    )

    return train_ds, test_ds, train_df, test_df
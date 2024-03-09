from torch.utils.data import Dataset
import torch

class Score_DS(Dataset):
    def __init__(self, tokenizer, max_length, questions, options, scores, country_ids):
        self.tokenizer = tokenizer
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

        prompt = question + self.tokenizer.sep_token + option
        prompt_enc = self.tokenizer(prompt, truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')
        
        return {
            'input_ids': prompt_enc['input_ids'].squeeze(),
            'attention_mask': prompt_enc['attention_mask'].squeeze(),
            'labels': torch.tensor(score, dtype=torch.float),
            'country_id': torch.tensor(country_id, dtype=torch.int64)
        }
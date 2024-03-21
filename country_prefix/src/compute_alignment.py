from transformers import AutoModelForSequenceClassification, AutoTokenizer
import sys
import pandas as pd
import numpy as np
from scipy.spatial import distance
from tqdm import tqdm 
import json
import os

sys.path.append("train_utils")
from score_ds import get_train_test_ds

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x)
    return exp_x / sum_exp_x

model_checkpoint_path = "country_prefix2/checkpoints/roberta_rm/all_prm/checkpoint-38190"

base_model = "FacebookAI/roberta-base"
data_dir = "country_prefix2/data"

model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint_path)
model = model.eval()

print(model)

# tokenizer = AutoTokenizer.from_pretrained(base_model)

# # countries = ["United States", "China"]
# with open("country_prefix2/src/countries.txt", mode="r") as f:
#     countries = f.read()
#     countries = countries.split("\n")[:-1]
#     print(countries)

# averaged_alignment = 0
# alignment_dict = dict()
# for country in countries:
#     # join_fn = lambda country, question, option: "{} {} {}".format(question, tokenizer.sep_token, option)
#     join_fn = lambda country, question, option: "[{}] {} {} {} {}".format(country, tokenizer.sep_token, question, tokenizer.sep_token, option)

#     train_ds, test_ds, train_df, test_df = get_train_test_ds(country, data_dir, tokenizer, join_fn, 256)
#     test_qs = test_df["question_id"].unique()

#     alignment = 0
#     counter = 0
#     for qid in tqdm(test_qs):
#         question_df = test_df[test_df["question_id"] == qid]

#         questions = list(question_df["question"])
#         options = list(question_df["option"])
#         gt_scores = np.array(list(question_df["score"]))

#         epsilon = np.full(len(gt_scores), 1e-5)
#         gt_scores = gt_scores + epsilon
#         gt_scores /= np.sum(gt_scores)

#         input_strs = list()
#         for i in range(len(questions)):
#             input_strs.append(join_fn(country, questions[i], options[i]))

#         tokenizer_out = tokenizer(input_strs, truncation=True, padding='max_length', max_length=256, return_tensors='pt')
#         input_ids = tokenizer_out["input_ids"]
#         attention_mask = tokenizer_out["attention_mask"]

#         preds = model(input_ids, attention_mask).logits.squeeze(-1)
#         preds = preds.detach().numpy()

#         preds[preds < 0] = 0
#         preds = preds + epsilon
#         preds /= np.sum(preds)
#         # preds = softmax(preds)

#         q_alignment = 1 - distance.jensenshannon(gt_scores, preds)
        
#         if q_alignment > 0 and q_alignment < 1:
#             alignment += q_alignment
#             counter += 1
#         else:
#             print(gt_scores, np.sum(gt_scores))
#             print(preds, np.sum(preds))

#     alignment /= counter
#     print(f"{country}: {alignment}")
#     averaged_alignment += alignment
#     alignment_dict[country] = alignment

#     with open(os.path.join("country_prefix2/src", "Roberta_All_Prefix_alignment.json"), "w") as json_file:
#         json.dump(alignment_dict, json_file, indent=4)

# averaged_alignment /= len(countries)
# print(f"Average alignment over all countries: {averaged_alignment}")
# alignment_dict["Average"] = averaged_alignment

# with open(os.path.join("country_prefix2/src", "Roberta_All_Prefix_alignment.json"), "w") as json_file:
#     json.dump(alignment_dict, json_file, indent=4)

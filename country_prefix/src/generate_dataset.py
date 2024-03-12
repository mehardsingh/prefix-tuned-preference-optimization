# python country_prefix2/utils/generate_dataset.py --save_dir country_prefix2/data

from datasets import load_dataset
import ast
import re
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split
import os
import argparse

def extract_curly_braces(string):
    # Using regular expression to find the substring within curly braces
    match = re.search(r'\{.*?\}', string)
    if match:
        return match.group(0)
    else:
        return None
    
def remove_parentheses(string):
    # Using regular expression to remove parentheses and their contents
    result = re.sub(r'\s*\([^()]*\)', '', string)
    return result

def map_to_str_dict(sample):    
    str_selections = sample["selections"]
    str_selections = extract_curly_braces(str_selections)
    str_selections = remove_parentheses(str_selections)
    str_selections = str_selections.strip()

    return {
        "question": sample["question"],
        "selections": str_selections,
        "options": sample["options"],
        "source": sample["source"]
    }

def get_all_countries(dataset):
    all_selections = dataset["selections"]
    all_countries = list()
    for selection in all_selections:
        dict_selection = ast.literal_eval(selection)
        selection_countries = list(dict_selection.keys())
        all_countries += selection_countries

    return all_countries

def get_country2code(all_countries):
    counter = 0
    country2code = dict()

    all_countries = set(all_countries)
    for country in all_countries:
        country2code[country] = counter
        counter += 1

    return country2code

def save_dataset(dataset, country2code, save_dir):
    df_question_id = list()
    df_question = list()
    df_option = list()
    df_option_num = list()
    df_option_tot = list()
    df_country = list()
    df_country_id = list()
    df_score = list()
    df_source = list()

    for i in tqdm(range(len(dataset))):
        question = dataset[i]["question"]
        selections = dataset[i]["selections"]
        options = dataset[i]["options"]
        source = dataset[i]["source"]

        dict_selection = ast.literal_eval(selections)
        countries = list(dict_selection.keys())
        num_countries = len(countries)

        list_options = ast.literal_eval(options)
        num_options = len(list_options)

        for j in range(num_countries):
            country = countries[j]
            country_selections = dict_selection[country]

            for k in range(num_options):
                option = str(list_options[k])
                score = country_selections[k]
                
                df_question_id.append(i)
                df_question.append(question)
                df_option.append(option)
                df_option_num.append(k)
                df_option_tot.append(num_options)
                df_country.append(country)
                df_country_id.append(country2code[country])
                df_score.append(score)
                df_source.append(source)


    df_dict = {
        "question_id": df_question_id,
        "question": df_question,
        "option": df_option,
        "option_id": df_option_num,
        "num_options": df_option_tot,
        "country": df_country,
        "country_id": df_country_id,
        "score": df_score,
        "source": df_source
    }

    dataset_df = pd.DataFrame(df_dict)
    dataset_df.to_csv(os.path.join(save_dir, "dataset.csv"))
    return dataset_df

def save_splits(dataset_df, save_dir, test_size=0.2):
    train_df, test_df = train_test_split(dataset_df, test_size=test_size, random_state=42)
    train_df.to_csv(os.path.join(save_dir, "train_ds.csv"))
    test_df.to_csv(os.path.join(save_dir, "test_ds.csv"))
    return train_df, test_df

def main(args):
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    dataset = load_dataset("Anthropic/llm_global_opinions")["train"]
    dataset = dataset.map(map_to_str_dict, num_proc=10)

    all_countries = get_all_countries(dataset)
    country2code = get_country2code(all_countries)

    dataset_df = save_dataset(dataset, country2code, args.save_dir)
    train_df, test_df = save_splits(dataset_df, args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str)
    args = parser.parse_args()

    main(args)
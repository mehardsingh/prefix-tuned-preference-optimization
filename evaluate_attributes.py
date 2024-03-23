import pandas as pd
from collections import Counter
import numpy as np
from tqdm import tqdm
from scipy.spatial import distance
from scipy.stats import wasserstein_distance
import matplotlib.pyplot as plt
import os
import json

with open("attributes.txt", mode="r") as f:
    attributes = f.readlines()
    attributes = [a.strip() for a in attributes]

df = pd.read_csv("WVS_Cross-National_Wave_7_csv_v5_0.csv")
questions = [f"Q{i}" for i in range(1, 260)]

def get_alignment(dist1, dist2, metric):
    epsilon = 1e-5
    
    if metric == "js":
        return 1 - distance.jensenshannon(dist1+epsilon, dist2+epsilon)
    else:
        return 1 - wasserstein_distance(
            np.arange(dist1.shape[0]), 
            np.arange(dist2.shape[0]), 
            u_weights=dist1+epsilon, 
            v_weights=dist2+epsilon
        ) / (dist2.shape[0] - 1)

def get_alignment_matrix(att_dists, metric="js"):    
    alignment_matrix = np.zeros((att_dists.shape[0], att_dists.shape[0]))
    for i in range(att_dists.shape[0]):
        for j in range(att_dists.shape[0]):
            alignment_matrix[i][j] = get_alignment(att_dists[i], att_dists[j], metric)
            
    return alignment_matrix

def get_attribute_options(attribute):
    att_options = {a for a in df[attribute] if a > 0}
    pbar = tqdm(questions, desc="Loading attribute options")
    for question in pbar:
        response_options = sorted(list({r for r in df[question] if r > 0}))
        question_df = df[df[question].isin(response_options)]
        atts_options4question = {a for a in question_df[attribute] if a > 0}
        att_options = att_options.intersection(atts_options4question)
        
    return att_options

def get_attribute_metrics(attribute):
    att_options = get_attribute_options(attribute)
    alignment_matrices = list()
    
    pbar = tqdm(questions, desc=attribute)
    for question in pbar:
        response_options = sorted(list({r for r in df[question] if r > 0}))
        question_df = df[df[question].isin(response_options)]
        att_prob_dists = list()

        for att_option in att_options:
            att_option_df = question_df[question_df[attribute] == att_option] # get the df associated with respondents of the particular attribute option
            if len(att_option_df) == 0:
                continue

            q_responses = list(att_option_df[question])
            response_counter = Counter(q_responses) # get the frequency of each question response option
            response_freq = dict(response_counter)

            for r in response_options:
                if not r in response_freq:
                    response_freq[r] = 0

            prob_dist = np.array([response_freq[key] for key in sorted(response_freq.keys())])
            prob_dist = prob_dist / np.sum(prob_dist) # get prob dist for respondents with att option on the question
            att_prob_dists.append(prob_dist)

        att_prob_dists = np.array(att_prob_dists)
        alignment_matrix = get_alignment_matrix(att_prob_dists, metric="emd") # compute NxN similarity matrix between N attribute options
        alignment_matrices.append(alignment_matrix)

    alignment_matrices = np.array(alignment_matrices)
    avg_alignment_matrix = np.mean(alignment_matrices, axis=0)

    mean = np.mean(avg_alignment_matrix[np.tril_indices(avg_alignment_matrix.shape[0], k=-1)])
    min_alignment = np.min(avg_alignment_matrix)
    
    return mean, min_alignment, avg_alignment_matrix

att_metric_dict = dict()

for i in range(0, len(attributes)):
    print(attributes[i])

    mean, min_alignment, avg_alignment_matrix = get_attribute_metrics(attributes[i])
    att_metric_dict[attributes[i]] = {
        "mean_alignment": mean,
        "min_alignment": min_alignment
    }

    with open("att_metric_dict.json", mode="w") as f:
        json.dump(att_metric_dict, f, indent=4)
    
    plt.imshow(avg_alignment_matrix, cmap='hot', vmin=0.65, vmax=1)
    plt.colorbar()

    plt.savefig(os.path.join("attribute_heatmaps", f"{attributes[i]}.png"))
    plt.clf()

# A = np.array([
#     [0.18287614, 0.14879468, 0.66832918], 
#     [0.29557522, 0.19292035, 0.51150442]
# ])
# print(get_alignment(A[0], A[1], metric="js"))
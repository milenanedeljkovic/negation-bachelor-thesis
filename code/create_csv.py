import csv
import torch
import sys

# This script incrementally merges dictionaries containing average representations of verbs
# and makes a new csv file at each step. It processed the first 10000 pages, writes them into a .csv file,
# then merges the next 10000 and writes the merged information into a new .csv files and so on.
# The first and last embeddings-avg file to process are given as arguments

def merge_dict(dict1, dict2):
    """Merges two dictionaries as created in script.py and written in files in embeddings-avg"""
    for key in dict2:
        if key in dict1:
            dict1[key] = [dict1[key][i] + dict2[key][i] for i in range(len(dict1[key]))]
        else:
            dict1[key] = dict2[key]
    return dict1


# the first and the last chunk of 10000 we want to process
first, last = sys.argv[1], sys.argv[2]
dict = {}
for i in range(first, last, 10000):
    next_dict = torch.load(f"embeddings-avg/embeddings-avg{i}")
    dict = merge_dict(dict, next_dict)
    with open(f"from{first}-from{i}.csv", "w") as file:
        writer = csv.writer(file)

        writer.writerow([f"First chunk: {first}", f"Last chunk: {i}"])
        writer.writerow(['lemma', 'total occ', 'affirmative', 'negated', 'perc negated', 'cosine'])

        for key in dict:
            total_occ = dict[key][2] + dict[key][3]
            writer.writerow([f"{key}", f"{total_occ}", f"{dict[key][3]}", f"{dict[key][2]}", f"{dict[key][2] / total_occ}",
                             f"{torch.nn.CosineSimilarity(dict[key][0], dict[key][1])}"])

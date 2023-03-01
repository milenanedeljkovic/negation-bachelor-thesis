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


dict = {}
cossim = torch.nn.CosineSimilarity(dim=0)

for i in range(110000, 180001, 10000):
    next_dict = torch.load(f"embeddings-avg/embeddings-avg{i}")
    dict = merge_dict(dict, next_dict)

for i in range(240000, 330001, 10000):
    next_dict = torch.load(f"embeddings-avg/embeddings-avg{i}")
    dict = merge_dict(dict, next_dict)

for i in range(350000, 360001, 10000):
    next_dict = torch.load(f"embeddings-avg/embeddings-avg{i}")
    dict = merge_dict(dict, next_dict)

for i in range(500000, 500001, 10000):
    next_dict = torch.load(f"embeddings-avg/embeddings-avg{i}")
    dict = merge_dict(dict, next_dict)

for i in range(520000, 750001, 10000):
    next_dict = torch.load(f"embeddings-avg/embeddings-avg{i}")
    dict = merge_dict(dict, next_dict)

with open(f"csv_files/1mars.csv", "w") as file:
    writer = csv.writer(file)

    writer.writerow([f"1 mars"])
    writer.writerow(['lemma', 'total occ', 'num non neg', 'num neg', 'perc neg', 'cos'])

    for key in dict:
        total_occ = dict[key][2] + dict[key][3]
        if type(dict[key][0]) == int or type(dict[key][1]) == int:
            # this means that there were no negated or no non-negated occurrences
            # the value in that case will be 0 and of type int
            cos_sim = 'undefined'
        else:
            cos_sim = cossim(dict[key][0] / dict[key][2], dict[key][1] / dict[key][3])
        writer.writerow([f"{key}", f"{total_occ}", f"{dict[key][3]}", f"{dict[key][2]}",
                         f"{dict[key][2] / total_occ}", f"{cos_sim}"])

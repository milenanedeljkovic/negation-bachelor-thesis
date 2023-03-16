import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
import spacy_conll
import spacy_stanza
import torch
from transformers import AutoModel, AutoTokenizer
import torch
from datetime import datetime
import conllu

def neg_cue_frequency(path: str):
    # reading the file from path
    f = open(path)
    dep_trees = conllu.parse_incr(f)

    num_not, num_never, num_more, num_longer = 0, 0, 0, 0

    for phrase in dep_trees:
        for word in phrase:
            if word['lemma'] in ["not", "cannot", "can't"]:
                if phrase[int(word['head']) - 1]['upos'] == "VERB":
                    num_not += 1
            elif word['lemma'] == "never":
                if phrase[int(word['head']) - 1]['upos'] == "VERB":
                    num_never += 1
            elif word['lemma'] == 'more' and word['id'] > 1 and phrase[word['id'] - 2]['lemma'] == 'no':
                if phrase[int(word['head']) - 1]['upos'] == "VERB":
                    num_more += 1
            elif word['lemma'] == 'longer' and word['id'] > 1 and phrase[word['id'] - 2]['lemma'] == 'no':
                if phrase[int(word['head']) - 1]['upos'] == "VERB":
                    num_longer += 1

    return num_not, num_never, num_more, num_longer


with torch.no_grad():
    lower, upper = int(sys.argv[1]), int(sys.argv[2])

    tot_not, tot_never, tot_more, tot_longer = 0, 0, 0, 0

    for first_page in range(lower, upper, 10000):
        print(f"{first_page} started at {datetime.now()}")
        dependency_trees = f"parsed/parsed{first_page}.conll"  # the file with parsed phrases

        if not os.path.isfile(dependency_trees):
            print(f"Not a file: {dependency_trees}")
            continue

        nn, nnv, nm, nl = neg_cue_frequency(dependency_trees)

        tot_not += nn
        tot_never += nnv
        tot_more += nm
        tot_longer += nl

    total = tot_not + tot_never + tot_more + tot_longer
    with open("cue-freq.txt", "w") as file:
        file.write(f"not: {tot_not} ({tot_not / total})\n")
        file.write(f"never: {tot_never} ({tot_never / total})\n")
        file.write(f"no more: {tot_more} ({tot_more / total})\n")
        file.write(f"no longer: {tot_longer} ({tot_longer / total})\n")



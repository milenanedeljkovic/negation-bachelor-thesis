import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import spacy_conll
import spacy_stanza
from datasets import load_dataset
import sys
from function_definitions import txt_to_conll


last_to_parse = sys.argv[1]

dataset = load_dataset("bigscience-data/roots_en_wikipedia", use_auth_token=True)

if last_to_parse >= len(dataset):
    raise ValueError(f"Cannot parse up to index {last_to_parse}, the last index of the dataset is {len(dataset) - 1}.")

with open("last_page_processed.txt", "r") as file:
    # the first line in this file contains the index of the last processed page in the corpus
    last_parsed = int(file.readline())

if last_to_parse <= last_parsed:
    raise ValueError(f"Already parsed up to index {last_parsed}.")

nlp = spacy_conll.init_parser("en", "stanza", parser_opts={"use_gpu": True, "verbose": False}, include_headers=True)

try:
    for i in range(last_parsed + 1, last_to_parse + 1):
        page_text = dataset[i]['text']
        with open("dependency_trees.conll", "a") as file:
            file.write(txt_to_conll(page_text, nlp))
except:
    raise RuntimeError("Parsing failed.")

with open("last_page_processed.txt", "w") as file:
    file.write(last_to_parse)

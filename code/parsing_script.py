import os
import spacy_conll
import spacy_stanza
from datasets import load_dataset
import sys
from datetime import datetime
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
def txt_to_conll(text: str, nlp):
    """Input:
    - text: the string we want to parse
    - nlp: stanza parser (initalized in the cell above)

    Output:
    - the dependency trees for each setnence in text,
      concatenated in a .conll format"""

    # text clean-up: we need to eliminate all \n and \t and not have more than one ' ' in a row anywhere
    # we do this by using string.split() method which splits by " ", \n and \t and concatenate all the pieces
    # excess spaces result in a wrong .conll that is undreadable afterwards
    text = " ".join(text.split())

    doc = nlp(text)
    return doc._.conll_str

with torch.no_grad():
    last_to_parse = int(sys.argv[1])

    dataset = load_dataset("bigscience-data/roots_en_wikipedia", use_auth_token=True)

    if last_to_parse >= len(dataset['train']):
        raise ValueError(f"Cannot parse up to index {last_to_parse}, the last index of the dataset is {len(dataset['train']) - 1}.")

    with open("last_page_processed.txt", "r") as file:
        # the first line in this file contains the index of the last processed page in the corpus
        last_parsed = int(file.readline())

    if last_to_parse <= last_parsed:
        raise ValueError(f"Already parsed up to index {last_parsed}.")

    nlp = spacy_conll.init_parser("en", "stanza", parser_opts={"use_gpu": True, "verbose": False}, include_headers=True)

    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Parsing Start Time =", current_time)

    try:
        for i in range(last_parsed + 1, last_to_parse + 1):
            page_text = dataset['train'][i]['text']
            with open("dependency_trees.conll", "a") as file:
                file.write(txt_to_conll(page_text, nlp))
    except:
        raise RuntimeError("Parsing failed.")


    now = datetime.now()

    current_time = now.strftime("%H:%M:%S")
    print("Parsing End Time =", current_time)
    print(f"Number of pages processed: {last_to_parse - last_parsed}")

    with open("last_page_processed.txt", "w") as file:
        file.write(last_to_parse)


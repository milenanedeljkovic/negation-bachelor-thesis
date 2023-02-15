import os
import spacy_conll
import spacy_stanza
from datasets import load_dataset
import sys
from datetime import datetime
import torch

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
    # if len(sys.argv) != 4:
    #     raise AttributeError(f"Script takes 3 arguments: the first and last index to parse and the output file.")
    # first_to_parse = int(sys.argv[1])
    # last_to_parse = int(sys.argv[2])
    # writefile = sys.argv[3]

    dataset = load_dataset("bigscience-data/roots_en_wikipedia", use_auth_token=True)

    # if last_to_parse > len(dataset['train']):
    #     raise ValueError(f"Cannot parse up to index {last_to_parse}, the length of the dataset is {len(dataset['train'])}.")

    nlp = spacy_conll.init_parser("en", "stanza", parser_opts={"use_gpu": True, "verbose": False}, include_headers=True)

    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"Parsing start time: {current_time}")

    for first_to_parse in range(80000, 2040001, 10000):
        last_to_parse = first_to_parse + 10000
        for i in range(first_to_parse, last_to_parse):
            page_text = dataset['train'][i]['text']
            with open(f"parsed/parsed{first_to_parse}.conll", "w") as file:
                file.write(txt_to_conll(page_text, nlp))
                current_time = datetime.now().strftime("%H:%M:%S")
                print(f"Pages [{first_to_parse}:{last_to_parse}] written to file at: {current_time}")

    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"Parsing end time: {current_time}")
    # print(f"Number of pages processed: {last_to_parse - first_to_parse}")


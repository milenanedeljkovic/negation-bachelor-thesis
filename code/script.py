import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys
import spacy_conll
import spacy_stanza
import torch
from transformers import AutoModel, AutoTokenizer
from function_definitions import txt_to_conll, get_contextual_embeddings
import torch


dataset = sys.argv[1]  # the file with parsed phrases

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained("roberta-base")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

if os.path.isfile("verb_embeddings"):
    verb_embeddings = torch.load("verb_embeddings")
else:
    verb_embeddings = {}

num_phrases, num_complex_phrases, num_negations, num_negations_in_dependent_clauses = 0, 0, 0, 0

embeddings = get_contextual_embeddings("current_page.conll", tokenizer, model, device)

for verb in embeddings:
    if verb not in verb_embeddings:
        verb_embeddings[verb] = embeddings[verb]
    else:
        verb_embeddings[verb] += embeddings[verb]  # this is addition of lists!

torch.save(verb_embeddings, "verb_embeddings")


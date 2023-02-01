import spacy_conll
import spacy_stanza
from spacy_conll import init_parser
import torch
from transformers import AutoModel, AutoTokenizer
from function_definitions import txt_to_conll, wiki_parsing, wiki_negated_clause_stats, get_verb_embeddings


nlp = init_parser("en", "stanza", parser_opts={"use_gpu": True, "verbose": False}, include_headers=True)
tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModel.from_pretrained('roberta-base')



import stanza
import spacy_conll
import spacy_stanza
from spacy_conll import init_parser
import timeit


nlp1 = init_parser("en", "stanza", parser_opts={"use_gpu": True, "verbose": False}, include_headers=True)
nlp2 = stanza.Pipeline('en', processors='tokenize,pos', use_gpu=True, pos_batch_size=3000)

t1 = timeit.timeit("nlp1(I like making clothes, but right now I am out of sewing thread.)", 20)
t2 = timeit.timeit("nlp2(I like making clothes, but right now I am out of sewing thread.)", 20)

print(f"With init_parser: {t1}s\nWith Pipeline: {t2}s")


import timeit

t1 = timeit.timeit("nlp1('I like making clothes, but right now I am out of sewing thread.')",
                   setup="import stanza\nimport spacy_conll\nimport spacy_stanza\n"
                         "nlp1 = spacy_conll.init_parser('en', 'stanza', parser_opts={'use_gpu': True, 'verbose': False},"
                         "include_headers=True)", number=20)
t2 = timeit.timeit("nlp2('I like making clothes, but right now I am out of sewing thread.')",
                   setup="import stanza\nimport spacy_conll\nimport spacy_stanza\n"
                         "nlp2 = stanza.Pipeline('en', processors='tokenize,pos', use_gpu=True, pos_batch_size=3000)",
                   number=20)

print(f"With init_parser: {t1}s\nWith Pipeline: {t2}s")


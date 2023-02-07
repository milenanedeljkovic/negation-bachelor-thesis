import os
import sys
import nltk
nltk.download()
from nltk import sent_tokenize


# this is dataset['train'] from dataset = load_dataset(...)
dataset = sys.argv[1]

with open("negation-phrases", "a") as file:
    for page in dataset:
        print(page)
        phrases = sent_tokenize(page.lower())
        for phrase in phrases:
            if any([s in phrase for s in ["n't", " not ", " never ", " not.", " not!", " not?", " never.", " never?",
                                          " never!", "cannot", "no more"]]):
                file.write(phrase)


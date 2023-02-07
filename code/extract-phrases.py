import nltk
nltk.download()
from nltk import sent_tokenize
from datasets import load_dataset

dataset = load_dataset("bigscience-data/roots_en_wikipedia", use_auth_token=True)

with open("negation-phrases", "a") as file:
    for page in dataset['train']:
        phrases = sent_tokenize(page['text'].lower())
        for phrase in phrases:
            if any([s in phrase for s in ["n't", " not ", " never ", " not.", " not!", " not?", " never.", " never?",
                                          " never!", "cannot", "no more"]]):
                file.write(phrase)


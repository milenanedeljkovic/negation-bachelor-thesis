import nltk
nltk.download()
from nltk import sent_tokenize
from datasets import load_dataset

dataset = load_dataset("bigscience-data/roots_en_wikipedia", use_auth_token=True)

print("here")

with open("negation-phrases", "a") as file:
    for page in dataset['train']:
        print("a page")
        phrases = sent_tokenize(page['text'].lower())
        for phrase in phrases:
            if any([s in phrase for s in ["not", "never", "no", "cannot", "n't"]]):
                file.write(phrase)



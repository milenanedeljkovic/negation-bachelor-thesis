import nltk
nltk.download()
from nltk import sent_tokenize
from datasets import load_dataset

dataset = load_dataset("bigscience-data/roots_en_wikipedia", use_auth_token=True)

i = 0
j = 0
with open("phrases-with-negation.txt", "a") as file:
    for page in dataset['train']:
        phrases = sent_tokenize(page['text'])
        for phrase in phrases:
            j += 1
            if any([s in phrase.lower() for s in ["not", "never", "no", "cannot", "n't", "no more"]]):
                i += 1
                file.write(f"{phrase}\n")


with open("stats.txt", "a") as file:
    file.write(f"Number of phrases with negation: {i} / {j} ({i / j}%)\n")


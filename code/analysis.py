import torch
import sys

average_embeddings = {}

lower, upper = sys.argv[1], sys.argv[2]

for i in range(lower, upper, 10000):
    emb = torch.load(f"embeddings-avg/embeddings-avg{i}")



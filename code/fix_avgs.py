import torch

for i in range(130000, 220001, 10000):
    embeddings = torch.load(f"embeddings/embeddings{i}")
    for key in embeddings:
        num_neg = len(embeddings[key][0])
        num_aff = len(embeddings[key][1])
        embeddings[key][0] = sum(embeddings[key][0])
        embeddings[key][1] = sum(embeddings[key][1])
        embeddings[key].append(num_neg)
        embeddings[key].append(num_aff)
    torch.save(embeddings, f"embeddings-avg/embeddings-avg{i}")
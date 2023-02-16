import torch

average_embeddings = {}

for i in range(80000, 1500001, 10000):
    emb = torch.load(f"embeddings/embeddings{i}")
    for key in emb:
        if key not in average_embeddings:
            average_embeddings[key] = sum(emb[key])
        else:
            average_embeddings[key] += sum(emb[key])

torch.save(average_embeddings, "average_embeddings")


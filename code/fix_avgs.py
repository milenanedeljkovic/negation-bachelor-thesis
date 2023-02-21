import torch

for i in range(110000, 220001, 10000):
    embeddings = torch.load(f"embeddings/embeddings{i}")
    for key in embeddings:
        embeddings[key][0] = sum(embeddings[key][0])
        embeddings[key][1] = sum(embeddings[key][1])
        embeddings.append(len(embeddings[key][0]))
        embeddings.append(len(embeddings[key][1]))
    torch.save(embeddings, f"embeddings-avg/embeddings-avg{i}")
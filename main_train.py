# main_train.py

import torch
from src.dataset import get_mnist
from src.shard import create_shards, create_slices
from src.train import train_shard
from src.config import S, R

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# reproducibility
torch.manual_seed(42)

dataset = get_mnist()

shards = create_shards(len(dataset), S)
shard_slices = create_slices(shards, R)
# ADD THIS BLOCK
index_map = {}

for shard_id, slices in enumerate(shard_slices):
    for slice_id, indices in enumerate(slices):
        for idx in indices:
            index_map[int(idx)] = (shard_id, slice_id)

torch.save({
    "shard_slices": shard_slices,
    "index_map": index_map
}, "shard_mapping.pt")

for shard_id, slices in enumerate(shard_slices):
    print(f"Training shard {shard_id}")
    train_shard(shard_id, slices, dataset, device)

# convert tensors → lists
# mapping = {
#     "shard_slices": [
#         [slice_indices.tolist() for slice_indices in shard]
#         for shard in shard_slices
#     ]
# }

# torch.save(mapping, "shard_mapping.pt")
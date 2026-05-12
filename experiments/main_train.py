# experiments/main_train.py

import torch

from src.dataset import get_mnist

from src.shard import (
    create_shards,
    create_slices
)

from src.train import train_shard

from src.config import (
    S,
    R
)


# ---------------- DEVICE ----------------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------- REPRODUCIBILITY ----------------

torch.manual_seed(42)


# ---------------- LOAD DATASET ----------------

dataset = get_mnist()


# ---------------- CREATE SHARDS ----------------

shards = create_shards(
    len(dataset),
    S
)

shard_slices = create_slices(
    shards,
    R
)


# ---------------- CREATE INDEX MAP ----------------

index_map = {}

for shard_id, slices in enumerate(shard_slices):

    for slice_id, indices in enumerate(slices):

        for idx in indices:

            index_map[int(idx)] = (
                shard_id,
                slice_id
            )


# ---------------- SAVE MAPPING ----------------

torch.save(
    {
        "shard_slices": shard_slices,
        "index_map": index_map
    },
    "shard_mapping.pt"
)

print("\nShard mapping saved")


# ---------------- TRAIN SHARDS ----------------

for shard_id, slices in enumerate(shard_slices):

    print(
        f"\nTraining shard {shard_id}"
    )

    train_shard(
        shard_id,
        slices,
        dataset,
        device
    )


print("\nSISA training completed")
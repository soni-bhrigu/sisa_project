# src/train.py

import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset

from src.model import Net

from src.config import (
    BASE_MODEL_DIR,
    BATCH_SIZE,
    EPOCHS_PER_SLICE,
    LEARNING_RATE
)


def train_shard(shard_id, slices, dataset, device):
    """
    Train one SISA shard incrementally slice-by-slice.
    """

    model = Net().to(device)

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    shard_path = f"{BASE_MODEL_DIR}/shard_{shard_id}"

    os.makedirs(shard_path, exist_ok=True)

    cumulative_indices = []

    for slice_id, slice_indices in enumerate(slices):

        # add current slice data
        cumulative_indices.extend(
            [int(idx) for idx in slice_indices]
        )

        subset = Subset(
            dataset,
            cumulative_indices
        )

        loader = DataLoader(
            subset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        # incremental training
        for epoch in range(EPOCHS_PER_SLICE):

            for x, y in loader:

                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                logits = model(x)

                loss = criterion(logits, y)

                loss.backward()

                optimizer.step()

        # save checkpoint
        torch.save(
            model.state_dict(),
            f"{shard_path}/slice_{slice_id}.pt"
        )

        print(
            f"Shard {shard_id} | "
            f"Slice {slice_id} trained"
        )

    print(f"\nShard {shard_id} training completed")
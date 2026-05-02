import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.model import Net
from src.config import MODEL_DIR, BATCH_SIZE, EPOCHS_PER_SLICE


def train_shard(shard_id, slices, dataset, device):
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    shard_path = f"{MODEL_DIR}/shard_{shard_id}"
    os.makedirs(shard_path, exist_ok=True)

    cumulative_indices = []

    for r, slice_indices in enumerate(slices):
        cumulative_indices.extend(slice_indices.tolist())

        subset = Subset(dataset, cumulative_indices)
        loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(EPOCHS_PER_SLICE):
            for x, y in loader:
                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()
                loss = criterion(model(x), y)
                loss.backward()
                optimizer.step()

        # checkpoint
        torch.save(
            model.state_dict(),
            f"{shard_path}/slice_{r}.pt"
        )
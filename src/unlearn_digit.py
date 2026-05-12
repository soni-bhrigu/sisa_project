# src/unlearn_digit.py

import os

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, Subset

from src.model import Net

from src.config import (
    BASE_MODEL_DIR,
    UNLEARN_SHARDED_MODEL_DIR,
    BATCH_SIZE,
    EPOCHS_PER_SLICE,
    LEARNING_RATE,
    R
)


def unlearn_digit_from_shard(shard_id, digit, dataset, shard_slices, device):
    """
    Remove all instances of a digit from one shard,
    retrain incrementally,
    and save into unlearn directory.
    """

    print(
        f"\nUnlearning digit {digit} "
        f"from shard {shard_id}"
    )

    # ---------------- LOAD BASE MODEL ----------------

    model = Net().to(device)

    model.load_state_dict(
        torch.load(
            f"{BASE_MODEL_DIR}/"
            f"shard_{shard_id}/"
            f"slice_{0}.pt"
        )
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    criterion = nn.CrossEntropyLoss()

    # ---------------- CREATE OUTPUT DIR ----------------

    shard_output_path = (
        f"{UNLEARN_SHARDED_MODEL_DIR}/"
        f"shard_{shard_id}"
    )

    os.makedirs(
        shard_output_path,
        exist_ok=True
    )

    # ---------------- RETRAIN ----------------

    cumulative_indices = []

    for slice_id, indices in enumerate(
        shard_slices[shard_id]
    ):

        filtered_indices = []

        for idx in indices:

            idx = int(idx)

            _, label = dataset[idx]

            # REMOVE TARGET DIGIT
            if label == digit:
                continue

            filtered_indices.append(idx)

        cumulative_indices.extend(filtered_indices)

        subset = Subset(
            dataset,
            cumulative_indices
        )

        loader = DataLoader(
            subset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        for epoch in range(EPOCHS_PER_SLICE):

            for x, y in loader:

                x = x.to(device)
                y = y.to(device)

                optimizer.zero_grad()

                logits = model(x)

                loss = criterion(logits, y)

                loss.backward()

                optimizer.step()

        # SAVE UNLEARNED CHECKPOINT
        torch.save(
            model.state_dict(),
            f"{shard_output_path}/"
            f"slice_{slice_id}.pt"
        )

        print(
            f"Shard {shard_id} | "
            f"Slice {slice_id} unlearned"
        )

    print(
        f"\nDigit {digit} removed "
        f"from shard {shard_id}"
    )
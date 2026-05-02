import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from src.model import Net
from src.config import MODEL_DIR, BATCH_SIZE, EPOCHS_PER_SLICE


# NEW: use direct lookup instead of looping
def find_location(data_index, index_map):
    return index_map.get(data_index, (None, None))


def unlearn(data_index, dataset, device):
    # load mapping
    mapping = torch.load("shard_mapping.pt")
    shard_slices = mapping["shard_slices"]
    index_map = mapping["index_map"]

    # fast lookup
    shard_id, slice_id = find_location(data_index, index_map)

    if shard_id is None:
        print("Data point not found")
        return

    print(f"Unlearning index {data_index} from shard {shard_id}, slice {slice_id}")

    # model setup
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    shard_path = f"{MODEL_DIR}/shard_{shard_id}"

    # load checkpoint BEFORE slice
    if slice_id > 0:
        checkpoint_path = f"{shard_path}/slice_{slice_id - 1}.pt"
        model.load_state_dict(torch.load(checkpoint_path))
    else:
        # IMPORTANT FIX: reinitialize cleanly
        model = Net().to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    # retrain from slice_id onward
    cumulative_indices = []

    for r, indices in enumerate(shard_slices[shard_id]):

        # convert once (handles tensor or list)
        indices_list = [int(i) for i in indices]

        if r < slice_id:
            cumulative_indices.extend(indices_list)

        else:
            # remove the target data point
            filtered = [idx for idx in indices_list if idx != data_index]
            cumulative_indices.extend(filtered)

            subset = Subset(dataset, cumulative_indices)
            loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

            for epoch in range(EPOCHS_PER_SLICE):
                for x, y in loader:
                    x, y = x.to(device), y.to(device)

                    optimizer.zero_grad()
                    loss = criterion(model(x), y)
                    loss.backward()
                    optimizer.step()

            # overwrite checkpoint
            torch.save(
                model.state_dict(),
                f"{shard_path}/slice_{r}.pt"
            )

    print("Unlearning completed")
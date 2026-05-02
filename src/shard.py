import torch

def create_shards(dataset_size, S):
    indices = torch.randperm(dataset_size)
    shard_size = dataset_size // S

    shards = []
    for i in range(S):
        start = i * shard_size
        end = (i + 1) * shard_size
        shards.append(indices[start:end])

    return shards


def create_slices(shards, R):
    shard_slices = []

    for shard in shards:
        slice_size = len(shard) // R
        slices = []

        for r in range(R):
            start = r * slice_size
            end = (r + 1) * slice_size
            slices.append(shard[start:end])

        shard_slices.append(slices)

    return shard_slices
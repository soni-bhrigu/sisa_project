# experiments/main_visualize_unlearning.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt

from src.dataset import get_mnist

from src.model import Net

from src.evaluate import (
    predict_full_distribution
)

from src.unlearn_digit import (
    unlearn_digit_from_shard
)

from src.config import (
    R,
    BASE_MODEL_DIR,
    UNLEARN_SHARDED_MODEL_DIR
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

dataset = get_mnist()

mapping = torch.load(
    "shard_mapping.pt"
)

shard_slices = mapping["shard_slices"]

digit = 5
target_shard = 0


# ---------------- FIND SAMPLE IMAGE ----------------

sample_index = None

for i in range(len(dataset)):

    _, label = dataset[i]

    if label == digit:
        sample_index = i
        break

image, label = dataset[sample_index]


# ---------------- BEFORE MODEL ----------------

model_before = Net().to(device)

model_before.load_state_dict(
    torch.load(
        f"{BASE_MODEL_DIR}/"
        f"shard_{target_shard}/"
        f"slice_{R - 1}.pt"
    )
)

(
    pred_before,
    conf_before,
    probs_before
) = predict_full_distribution(
    model_before,
    image,
    device
)


# ---------------- UNLEARN ----------------

unlearn_digit_from_shard(
    target_shard,
    digit,
    dataset,
    shard_slices,
    device
)


# ---------------- AFTER MODEL ----------------

model_after = Net().to(device)

model_after.load_state_dict(
    torch.load(
        f"{UNLEARN_SHARDED_MODEL_DIR}/"
        f"shard_{target_shard}/"
        f"slice_{R - 1}.pt"
    )
)

(
    pred_after,
    conf_after,
    probs_after
) = predict_full_distribution(
    model_after,
    image,
    device
)


# ---------------- VISUALIZATION ----------------

plt.figure(figsize=(12, 4))


# ORIGINAL IMAGE
plt.subplot(1, 3, 1)

plt.imshow(
    image.squeeze(),
    cmap="gray"
)

plt.title(
    f"True Digit: {digit}"
)

plt.axis("off")


# BEFORE UNLEARNING
plt.subplot(1, 3, 2)

plt.bar(
    range(10),
    probs_before
)

plt.title(
    f"Before\n"
    f"Pred={pred_before}\n"
    f"Conf={conf_before:.4f}"
)

plt.xlabel("Digit Class")

plt.ylabel("Probability")


# AFTER UNLEARNING
plt.subplot(1, 3, 3)

plt.bar(
    range(10),
    probs_after
)

plt.title(
    f"After\n"
    f"Pred={pred_after}\n"
    f"Conf={conf_after:.4f}"
)

plt.xlabel("Digit Class")

plt.ylabel("Probability")


plt.tight_layout()

plt.show()
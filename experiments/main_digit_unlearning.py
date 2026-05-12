# experiments/main_digit_unlearning.py

import torch

from src.dataset import get_mnist

from src.model import Net

from src.evaluate import (
    evaluate_digit_confidence
)

from src.unlearn_digit import (
    unlearn_digit_from_shard
)

from src.config import (
    R,
    BASE_MODEL_DIR,
    UNLEARN_SHARDED_MODEL_DIR
)


# ---------------- DEVICE ----------------

device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ---------------- LOAD DATASET ----------------

dataset = get_mnist()

mapping = torch.load(
    "shard_mapping.pt"
)

shard_slices = mapping["shard_slices"]


# ---------------- EXPERIMENT CONFIG ----------------

digit = 5

target_shard = 0


# ---------------- LOAD BASE MODEL ----------------

model_before = Net().to(device)

model_before.load_state_dict(
    torch.load(
        f"{BASE_MODEL_DIR}/"
        f"shard_{target_shard}/"
        f"slice_{R - 1}.pt"
    )
)


# ---------------- EVALUATE BEFORE ----------------

before_confidence = (
    evaluate_digit_confidence(
        model_before,
        dataset,
        digit,
        device
    )
)

print("\nBEFORE UNLEARNING")

print(
    f"Average confidence "
    f"for digit {digit}: "
    f"{before_confidence:.4f}"
)


# ---------------- UNLEARN ----------------

unlearn_digit_from_shard(
    target_shard,
    digit,
    dataset,
    shard_slices,
    device
)


# ---------------- LOAD UNLEARNED MODEL ----------------

model_after = Net().to(device)

model_after.load_state_dict(
    torch.load(
        f"{UNLEARN_SHARDED_MODEL_DIR}/"
        f"shard_{target_shard}/"
        f"slice_{R - 1}.pt"
    )
)


# ---------------- EVALUATE AFTER ----------------

after_confidence = (
    evaluate_digit_confidence(
        model_after,
        dataset,
        digit,
        device
    )
)

print("\nAFTER UNLEARNING")

print(
    f"Average confidence "
    f"for digit {digit}: "
    f"{after_confidence:.4f}"
)


# ---------------- CONFIDENCE DROP ----------------

confidence_drop = (
    before_confidence -
    after_confidence
)

print("\nFORGETTING EFFECT")

print(
    f"Confidence Drop: "
    f"{confidence_drop:.4f}"
)
# src/aggregation.py

import os

import torch
import torch.nn.functional as F

from src.model import Net

from src.config import (
    S,
    R,
    BASE_MODEL_DIR,
    UNLEARN_SHARDED_MODEL_DIR
)


def ensemble_predict(
    image,
    device,
    use_unlearned=False
):
    """
    Aggregate predictions across all shards
    using probability averaging.
    """

    image = image.unsqueeze(0).to(device)

    all_probs = []

    for shard_id in range(S):

        model = Net().to(device)

        # ---------------- CHOOSE MODEL PATH ----------------

        unlearned_path = (
            f"{UNLEARN_SHARDED_MODEL_DIR}/"
            f"shard_{shard_id}/"
            f"slice_{R - 1}.pt"
        )

        base_path = (
            f"{BASE_MODEL_DIR}/"
            f"shard_{shard_id}/"
            f"slice_{R - 1}.pt"
        )

        # use unlearned shard if available
        if use_unlearned and os.path.exists(unlearned_path):

            model_path = unlearned_path

        else:
            model_path = base_path

        # ---------------- LOAD MODEL ----------------

        model.load_state_dict(
            torch.load(model_path)
        )

        model.eval()

        with torch.no_grad():

            logits = model(image)

            probs = F.softmax(
                logits,
                dim=1
            )

            all_probs.append(probs)

    # ---------------- AGGREGATE ----------------

    mean_probs = torch.mean(
        torch.stack(all_probs),
        dim=0
    )

    pred = torch.argmax(
        mean_probs,
        dim=1
    ).item()

    confidence = mean_probs[0][pred].item()

    return (
        pred,
        confidence,
        mean_probs.cpu().numpy()[0]
    )
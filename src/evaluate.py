# src/evaluate.py

import torch
import torch.nn.functional as F


def predict(
    model,
    image,
    device
):
    """
    Predict class and confidence for one image.
    """

    model.eval()

    with torch.no_grad():

        image = image.unsqueeze(0).to(device)

        logits = model(image)

        probs = F.softmax(
            logits,
            dim=1
        )

        pred = torch.argmax(
            probs,
            dim=1
        ).item()

        confidence = probs[0][pred].item()

    return pred, confidence


def predict_full_distribution(
    model,
    image,
    device
):
    """
    Return full probability distribution.
    """

    model.eval()

    with torch.no_grad():

        image = image.unsqueeze(0).to(device)

        logits = model(image)

        probs = F.softmax(
            logits,
            dim=1
        )

        pred = torch.argmax(
            probs,
            dim=1
        ).item()

        confidence = probs[0][pred].item()

    return (
        pred,
        confidence,
        probs.cpu().numpy()[0]
    )


def evaluate_digit_confidence(
    model,
    dataset,
    digit,
    device
):
    """
    Average confidence for one digit class
    across entire dataset.
    """

    model.eval()

    confidences = []

    with torch.no_grad():

        for image, label in dataset:

            if label != digit:
                continue

            image = image.unsqueeze(0).to(device)

            logits = model(image)

            probs = F.softmax(
                logits,
                dim=1
            )

            confidence = probs[0][digit].item()

            confidences.append(confidence)

    avg_confidence = (
        sum(confidences) / len(confidences)
    )

    return avg_confidence
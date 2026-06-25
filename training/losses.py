import torch
import torch.nn.functional as F


def compute_loss(outputs, labels):

    # Classification
    classification_loss = F.cross_entropy(
        outputs["scores"],
        labels,
    )

    collapse_loss = outputs["collapse_loss"]

    if outputs["validator_potential"] is not None:
        potential_loss = outputs["validator_potential"].pow(2).mean()
    else:
        potential_loss = torch.tensor(
            0.0,
            device=labels.device,
        )

    if outputs["quantum_energy"] is not None:
        energy_loss = outputs["quantum_energy"].pow(2).mean()
    else:
        energy_loss = torch.tensor(
            0.0,
            device=labels.device,
        )

    validator_loss = torch.tensor(
        0.0,
        device=labels.device,
    )

    validator = outputs.get("validator")

    if validator is not None and validator.get("relevance_target") is not None:

        validator_loss = F.binary_cross_entropy_with_logits(
            validator["relevance_logits"],
            validator["relevance_target"],
        )

    total_loss = (
        classification_loss
        + 0.02 * collapse_loss
        + 0.001 * potential_loss
        + 0.001 * energy_loss
        + 0.02 * validator_loss
    )

    if torch.isnan(total_loss):
        raise RuntimeError("NaN detected in total loss")

    return total_loss

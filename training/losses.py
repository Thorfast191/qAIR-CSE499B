import torch
import torch.nn.functional as F


def compute_loss(outputs, labels):

    # ========================================
    # Classification
    # ========================================

    classification_loss = F.cross_entropy(
        outputs["scores"],
        labels
    )

    # ========================================
    # Collapse
    # ========================================

    collapse_loss = outputs["collapse_loss"]

    # ========================================
    # Validator Potential Regularization
    # ========================================

    if outputs["validator_potential"] is not None:

        potential_loss = (
            outputs["validator_potential"]
            .pow(2)
            .mean()
        )

    else:

        potential_loss = torch.tensor(
            0.0,
            device=labels.device
        )

    # ========================================
    # Quantum Energy Regularization
    # ========================================

    if outputs["quantum_energy"] is not None:

        energy_loss = (
            outputs["quantum_energy"]
            .var(dim=1)
            .mean()
        )

    else:

        energy_loss = torch.tensor(
            0.0,
            device=labels.device
        )

    # ========================================
    # Total Loss
    # ========================================

    total_loss = (

        classification_loss

        + 0.10 * collapse_loss

        + 0.01 * potential_loss

        + 0.01 * energy_loss

    )

    return total_loss
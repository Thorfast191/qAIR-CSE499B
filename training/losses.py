import torch
import torch.nn.functional as F


def compute_loss(outputs, labels):

    ########################################################
    # Classification
    ########################################################

    classification_loss = F.cross_entropy(
        outputs["scores"],
        labels,
    )

    ########################################################
    # Margin Energy Loss
    ########################################################

    scores = outputs["scores"]

    correct = scores.gather(
        1,
        labels.unsqueeze(1),
    )

    margin = 1.0

    ranking_loss = F.relu(
        margin - correct + scores
    )

    ranking_loss.scatter_(
        1,
        labels.unsqueeze(1),
        0,
    )

    ranking_loss = ranking_loss.mean()

    ########################################################
    # Collapse
    ########################################################

    collapse_loss = outputs["collapse_loss"]

    ########################################################
    # Validator Potential
    ########################################################

    if outputs["validator_potential"] is not None:

        potential_loss = outputs[
            "validator_potential"
        ].pow(2).mean()

    else:

        potential_loss = torch.tensor(
            0.0,
            device=labels.device,
        )

    ########################################################
    # Quantum Energy
    ########################################################

    if outputs["quantum_energy"] is not None:

        quantum_loss = outputs[
            "quantum_energy"
        ].pow(2).mean()

    else:

        quantum_loss = torch.tensor(
            0.0,
            device=labels.device,
        )

    ########################################################
    # Validator BCE
    ########################################################

    validator_loss = torch.tensor(
        0.0,
        device=labels.device,
    )

    validator = outputs.get("validator")

    if (

        validator is not None

        and

        validator.get("relevance_target") is not None

    ):

        validator_loss = F.binary_cross_entropy_with_logits(

            validator["relevance_logits"],

            validator["relevance_target"],

        )

    ########################################################
    # Hypothesis Diversity
    ########################################################

    probs = outputs["collapse_probs"]

    diversity_loss = -(

        probs * torch.log(probs + 1e-8)

    ).sum(dim=1).mean()

    ########################################################
    # Total
    ########################################################

    total_loss = (

        1.0 * classification_loss

        +

        0.40 * ranking_loss

        +

        0.10 * collapse_loss

        +

        0.10 * validator_loss

        +

        0.05 * potential_loss

        +

        0.05 * quantum_loss

        +

        0.02 * diversity_loss

    )

    if torch.isnan(total_loss):

        raise RuntimeError("NaN detected in loss")

    return total_loss
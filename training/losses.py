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
    # Margin Ranking Loss
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

    mask = torch.ones_like(
        ranking_loss,
        dtype=torch.bool,
    )

    mask.scatter_(
        1,
        labels.unsqueeze(1),
        False,
    )

    ranking_loss = ranking_loss.masked_select(mask).mean()

    ########################################################
    # Collapse Loss
    ########################################################

    collapse_loss = outputs["collapse_loss"]

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
    # Entropy Regularization
    # Match CollapseController target entropy
    ########################################################

    probs = outputs["collapse_probs"]

    entropy = -(
        probs * torch.log(probs + 1e-8)
    ).sum(dim=1).mean()

    target_entropy = 1.2

    entropy_loss = (
        entropy - target_entropy
    ).pow(2)

    ########################################################
    # Total Loss
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

        0.02 * entropy_loss

    )

    ########################################################
    # NaN Guard
    ########################################################

    if not torch.isfinite(total_loss):

        raise RuntimeError(
            "Loss became NaN/Inf"
        )

    return total_loss
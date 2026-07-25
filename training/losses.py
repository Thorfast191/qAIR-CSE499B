import torch
import torch.nn.functional as F


def compute_loss(outputs, labels, diversity_weight=0.02):
    """
    Fixed in the audit
    ------------------
    * The entropy target was applied TWICE -- once inside
      CollapseController.collapse_loss and again here -- and it was
      anti-calibration in the first place: pinning entropy at 0.5 nats
      forces high confidence whether or not confidence is warranted. On
      a 4-way task where the model is right ~34% of the time the correct
      posterior is often near-uniform. Both copies are gone;
      cross-entropy calibrates the distribution.

    * CollapseController also ADDED `diversity` (variance of energy
      across hypotheses) and `spread` (max - min) to the loss with
      positive coefficients. Both are globally minimized at exactly zero,
      reached when every hypothesis has identical energy -- the loss was
      paying the model to collapse the superposition it exists to
      maintain. Training duly drove them to 8.2e-09 and 1.4e-08. The sign
      is now REVERSED and saturating: -tanh(diversity) rewards keeping
      hypotheses distinguishable but cannot be farmed indefinitely.

    * Padded options are excluded from both cross-entropy and the
      ranking loss.
    """

    scores = outputs["scores"]

    O_mask = outputs.get("O_mask")

    ########################################################
    # Classification
    #
    # full_model already masks padded options to dtype-min, so
    # cross_entropy assigns them ~zero probability.
    ########################################################

    classification_loss = F.cross_entropy(
        scores,
        labels,
    )

    ########################################################
    # Margin Ranking Loss
    ########################################################

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

    if O_mask is not None:
        # A padded option sits at dtype-min, which would otherwise make
        # its hinge term trivially zero and dilute the mean.
        mask = mask & O_mask

    ranking_loss = ranking_loss.masked_select(mask).mean()

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

        logits = validator["relevance_logits"]
        target = validator["relevance_target"]

        bce = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )

        H_mask = outputs.get("H_mask")

        if O_mask is not None or H_mask is not None:
            B, K, N = bce.shape
            m = torch.ones_like(bce, dtype=torch.bool)
            if H_mask is not None:
                m = m & H_mask.view(B, K, 1)
            if O_mask is not None:
                m = m & O_mask.view(B, 1, N)
            validator_loss = bce.masked_select(m).mean()
        else:
            validator_loss = bce.mean()

    ########################################################
    # Hypothesis diversity REWARD (see docstring)
    ########################################################

    diversity = outputs.get("diversity")

    diversity_loss = torch.tensor(0.0, device=labels.device)

    if diversity is not None:
        diversity_loss = -torch.tanh(diversity)

    ########################################################
    # Total Loss
    ########################################################

    total_loss = (

        1.0 * classification_loss

        +

        0.40 * ranking_loss

        +

        0.10 * validator_loss

        +

        diversity_weight * diversity_loss

    )

    ########################################################
    # NaN Guard
    ########################################################

    if not torch.isfinite(total_loss):

        raise RuntimeError(
            "Loss became NaN/Inf"
        )

    return total_loss

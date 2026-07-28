import torch
import torch.nn.functional as F

from config import LABEL_SMOOTHING


def compute_loss(outputs, labels, diversity_weight=0.02,
                 label_smoothing=LABEL_SMOOTHING):
    """
    `outputs["scores"]` is a normalized log-probability vector in v45
    (models/interference.py returns log |sum_k c_k v_kn|^2 after
    normalization, and full_model re-applies log_softmax after adding the
    LLM prior). F.cross_entropy applies log_softmax again, which is a
    no-op on an already-normalized vector -- log_softmax(log p) = log p
    exactly -- so this stays correct without special-casing.

    Fixed in the v44 audit, kept
    ----------------------------
    * The entropy target was applied TWICE -- inside the collapse
      controller and again here -- and was anti-calibration in the first
      place: pinning entropy at 0.5 nats forces high confidence whether
      or not it is warranted. On a 4-way task where the model is right
      ~34% of the time, a near-uniform posterior is often the correct
      one. Both copies are gone; cross-entropy calibrates the
      distribution.

    * The collapse controller ADDED `diversity` (variance of energy
      across hypotheses) and `spread` (max - min) to the loss with
      positive coefficients. Both are globally minimized at exactly zero,
      reached when every hypothesis has identical energy -- the loss was
      paying the model to collapse the superposition it exists to
      maintain, and training drove them to 8.2e-09 and 1.4e-08. The sign
      is reversed and saturating here: -tanh(diversity) rewards keeping
      hypotheses distinguishable but cannot be farmed indefinitely.

    * Padded options are excluded from both cross-entropy and the ranking
      loss.

    Deliberately NOT a loss term in v45
    -----------------------------------
    `interference_ratio`. Rewarding it would guarantee the number the
    quantum claim is tested by, which is exactly the mistake the
    diversity/spread terms made in reverse. It is reported every epoch
    and left to the data.
    """

    scores = outputs["scores"]

    O_mask = outputs.get("O_mask")

    ########################################################
    # Classification
    #
    # Label smoothing is applied over the VALID options only. Using
    # F.cross_entropy(..., label_smoothing=eps) would spread eps/N across
    # all N columns including the zero-padded ones, so a 3-option
    # question in a 5-option batch would be charged for probability mass
    # the model is explicitly forbidden to place. ARC has both 3- and
    # 5-option questions, so this is not hypothetical.
    #
    # `scores` is already a normalized log-probability vector, so
    # log_softmax here is a no-op that costs nothing and keeps this
    # correct if a caller ever passes raw logits.
    ########################################################

    log_probs = F.log_softmax(scores, dim=1)

    valid = (
        O_mask.to(log_probs.dtype)
        if O_mask is not None
        else torch.ones_like(log_probs)
    )

    n_valid = valid.sum(dim=1, keepdim=True).clamp(min=1.0)

    target = torch.zeros_like(log_probs)
    target.scatter_(1, labels.unsqueeze(1), 1.0)

    target = (1.0 - label_smoothing) * target + label_smoothing * valid / n_valid

    classification_loss = -(target * log_probs).sum(dim=1).mean()

    ########################################################
    # Margin ranking loss
    #
    # Operates on log-probabilities, so a margin of 1.0 asks the correct
    # option to be ~e times more probable than each distractor.
    ########################################################

    correct = scores.gather(1, labels.unsqueeze(1))

    margin = 1.0

    ranking_loss = F.relu(margin - correct + scores)

    mask = torch.ones_like(ranking_loss, dtype=torch.bool)

    mask.scatter_(1, labels.unsqueeze(1), False)

    if O_mask is not None:
        # A padded option sits at dtype-min, which would otherwise make
        # its hinge term trivially zero and dilute the mean.
        mask = mask & O_mask

    ranking_loss = ranking_loss.masked_select(mask).mean()

    ########################################################
    # Validator BCE
    ########################################################

    validator_loss = torch.tensor(0.0, device=labels.device)

    validator = outputs.get("validator")

    if validator is not None and validator.get("relevance_target") is not None:

        logits = validator["relevance_logits"]
        target = validator["relevance_target"]

        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

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
    # Total
    ########################################################

    total_loss = (
        1.0 * classification_loss
        + 0.40 * ranking_loss
        + 0.10 * validator_loss
        + diversity_weight * diversity_loss
    )

    if not torch.isfinite(total_loss):
        raise RuntimeError("Loss became NaN/Inf")

    return total_loss

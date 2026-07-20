"""
FIXED LOSS FUNCTION
- Removed redundant collapse_loss (already optimized in CollapseController)
- Removed redundant entropy_loss (conflicts with collapse controller target)
- Removed validator_loss (not properly supervised for multi-choice task)
- Reduced ranking_loss weight and margin
- Now: Simple, focused optimization on classification + gentle ranking
"""

import torch
import torch.nn.functional as F


def compute_loss(outputs, labels):

    ########################################################
    # PRIMARY OBJECTIVE: Classification
    ########################################################

    classification_loss = F.cross_entropy(
        outputs["scores"],
        labels,
    )

    ########################################################
    # SECONDARY OBJECTIVE: Ranking Loss (gentle)
    # Encourages correct answer to be higher than incorrect ones
    # with a small margin, but doesn't dominate optimization
    ########################################################

    scores = outputs["scores"]

    correct = scores.gather(
        1,
        labels.unsqueeze(1),
    )

    margin = 0.3  # REDUCED from 1.0 to reduce competition with classification_loss

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
    # REMOVED: collapse_loss
    # REASON: CollapseController already optimizes entropy
    #         with its own loss function (collapse_loss).
    #         Including it here creates double penalty and
    #         conflicting gradients.
    ########################################################

    ########################################################
    # REMOVED: entropy_loss
    # REASON: Both losses target entropy=0.5, creating
    #         redundant objectives that prevent smooth
    #         convergence.
    ########################################################

    ########################################################
    # REMOVED: validator_loss
    # REASON: Validator relevance target only set when
    #         label is available. Binary BCE doesn't align
    #         well with multi-choice task. Validator provides
    #         too much noise without clear signal.
    ########################################################

    ########################################################
    # TOTAL LOSS: Simple and focused
    ########################################################

    total_loss = (
        1.0 * classification_loss
        +
        0.08 * ranking_loss  # REDUCED weight from 0.40
    )

    ########################################################
    # NaN Guard
    ########################################################

    if not torch.isfinite(total_loss):

        raise RuntimeError(
            "Loss became NaN/Inf"
        )

    return total_loss

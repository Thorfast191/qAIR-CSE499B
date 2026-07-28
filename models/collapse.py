import torch
import torch.nn as nn
import torch.nn.functional as F


class CollapseController(nn.Module):
    """
    Adaptive per-sample temperature for the collapse, plus the collapse
    diagnostics.

    What changed in v45
    -------------------
    This module used to apply the Born rule itself: amplitude =
    exp(-E/2T), L2-normalize, square. models/validator.py applied the
    same pattern over a different axis, so the "Born rule" appeared twice
    in one forward pass -- and in both places it was, with real
    non-negative amplitudes, algebraically identical to softmax(-E/T)
    (verified to fp32 epsilon, 5.96e-08). Two copies of a softmax wearing
    a physics name.

    The amplitudes now live in exactly one place, models/interference.py,
    where they are COMPLEX and the squaring therefore produces genuine
    cross-terms. What is left here is the part that was actually doing
    something: a per-sample temperature read off the energy distribution,
    which controls how sharply the state collapses for this particular
    question, and the diagnostics.

    Fixed in the v44 audit, kept
    ----------------------------
    * `diversity` (variance of energy across hypotheses) and `spread`
      (max - min) were ADDED to the loss with positive coefficients. Both
      are globally minimized at exactly zero, attained when every
      hypothesis has identical energy -- the loss was paying the model to
      collapse the superposition it exists to maintain, and training duly
      drove them to 8.2e-09 and 1.4e-08. They are diagnostics here; the
      loss uses -tanh(diversity), a saturating reward.
    * The fixed entropy target (0.5 nats) was applied here AND in
      training/losses.py -- double-counted, and anti-calibration in the
      first place. Gone; cross-entropy calibrates the distribution.
    """

    def __init__(self):

        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )

        self.temperature = nn.Linear(32, 1)

        self.confidence = nn.Sequential(
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, energy, mask=None):
        """
        energy : (B, K) fused per-hypothesis energy
        mask   : (B, K) bool, True for real hypotheses

        Returns a per-sample temperature of shape (B, 1) for
        models/interference.py, and diagnostics on the raw energy.
        """

        raw_energy = energy

        z = self.encoder(torch.tanh(energy).unsqueeze(-1))

        if mask is not None:
            # Don't let padded hypotheses shift the pooled statistics the
            # temperature and confidence heads read.
            m = mask.unsqueeze(-1).to(z.dtype)
            z_global = (z * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            z_global = z.mean(dim=1)

        temperature = 0.3 + F.softplus(self.temperature(z_global))

        confidence = self.confidence(z_global).squeeze(-1)

        # ------------------------------------------------------------
        # Diagnostics (NOT loss terms -- see class docstring)
        # ------------------------------------------------------------

        if mask is not None:
            valid = mask.to(raw_energy.dtype)
            n = valid.sum(dim=1).clamp(min=1.0)
            mean_e = (raw_energy * valid).sum(dim=1) / n
            diversity = (
                ((raw_energy - mean_e.unsqueeze(-1)) ** 2) * valid
            ).sum(dim=1) / n
            spread = (
                raw_energy.masked_fill(~mask, float("-inf")).max(dim=1).values
                - raw_energy.masked_fill(~mask, float("inf")).min(dim=1).values
            )
        else:
            diversity = raw_energy.var(dim=1)
            spread = raw_energy.max(dim=1).values - raw_energy.min(dim=1).values

        return {
            "temperature": temperature,
            "confidence": confidence,
            "raw_energy": raw_energy,
            "diversity": diversity.mean(),
            "spread": spread.mean(),
        }

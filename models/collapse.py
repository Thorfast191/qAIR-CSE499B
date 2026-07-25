import torch
import torch.nn as nn
import torch.nn.functional as F


class CollapseController(nn.Module):
    """
    Adaptive Collapse Controller.

    Applies a temperature-scaled Born-rule step over the fused
    per-hypothesis energy to decide how much each hypothesis is trusted
    in the final answer.

    Honest note on the "Born rule" (see README)
    -------------------------------------------
    With real, non-negative amplitudes, amplitude = exp(-E/2T) followed
    by L2 normalization and squaring is *algebraically identical* to
    softmax(-E / T). Verified numerically to fp32 epsilon (5.96e-08).
    It is written in amplitude form because that is the object the
    project reasons about, and because it is the natural hook for the
    complex-amplitude extension (where cross-terms 2*Re(a_i conj(a_j))
    would make it genuinely non-classical) -- but as it stands it carries
    no expressive content beyond a softmax, and the README says so.

    Fixed in the audit
    ------------------
    * `diversity` and `spread` were previously ADDED to the loss with
      positive coefficients. Both are globally minimized at exactly zero,
      attained when every hypothesis has identical energy -- i.e. the
      loss was rewarding total hypothesis collapse in a model whose whole
      premise is keeping hypotheses distinct. Training drove them to
      8.2e-09 and 1.4e-08 respectively. They are now diagnostics only.
    * The fixed entropy target (0.5 nats) was applied here AND again in
      training/losses.py -- double-counted, and anti-calibration: it
      forced high confidence regardless of whether confidence was
      warranted. Removed; cross-entropy calibrates the distribution.
    """

    def __init__(self):

        super().__init__()

        ####################################################
        # Collapse encoder
        ####################################################

        self.encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Linear(32, 32),
            nn.GELU(),
        )

        ####################################################
        # Adaptive temperature
        ####################################################

        self.temperature = nn.Linear(32, 1)

        ####################################################
        # Confidence
        ####################################################

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
        """

        raw_energy = energy

        ####################################################
        # Normalize
        ####################################################

        energy = torch.tanh(energy)

        ####################################################
        # Encode energy distribution
        ####################################################

        z = self.encoder(
            energy.unsqueeze(-1)
        )

        if mask is not None:
            # Don't let padded hypotheses shift the pooled statistics the
            # temperature and confidence heads read.
            m = mask.unsqueeze(-1).to(z.dtype)
            z_global = (z * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        else:
            z_global = z.mean(dim=1)

        ####################################################
        # Adaptive parameters
        ####################################################

        temperature = (
            0.3
            +
            F.softplus(
                self.temperature(z_global)
            )
        )

        confidence = self.confidence(z_global).squeeze(-1)

        ####################################################
        # Born rule (numerically stabilized)
        #
        # amplitude ~ exp(-E / 2T), L2-normalized; probability =
        # amplitude^2. Equivalent to softmax(-E / T) -- see class
        # docstring. validator.py uses the identical convention; before
        # the audit it L1-normalized instead, which is not the Born rule
        # and gave it an effective temperature of T/2 for the same
        # nominal T.
        ####################################################

        log_amp = -energy / (2.0 * temperature)

        if mask is not None:
            log_amp = log_amp.masked_fill(~mask, float("-inf"))

        log_amp = log_amp - log_amp.max(
            dim=1,
            keepdim=True,
        ).values

        amplitude = torch.exp(log_amp)

        amplitude = amplitude / torch.sqrt(
            (amplitude ** 2).sum(dim=1, keepdim=True) + 1e-8
        )

        probabilities = amplitude.pow(2)

        probabilities = probabilities / (
            probabilities.sum(
                dim=1,
                keepdim=True,
            )
            + 1e-8
        )

        ####################################################
        # Diagnostics (NOT loss terms -- see class docstring)
        ####################################################

        entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=1)

        if mask is not None:
            valid = mask.to(raw_energy.dtype)
            n = valid.sum(dim=1).clamp(min=1.0)
            mean_e = (raw_energy * valid).sum(dim=1) / n
            diversity = ((raw_energy - mean_e.unsqueeze(-1)) ** 2 * valid).sum(dim=1) / n
            spread = (
                raw_energy.masked_fill(~mask, float("-inf")).max(dim=1).values
                - raw_energy.masked_fill(~mask, float("inf")).min(dim=1).values
            )
        else:
            diversity = raw_energy.var(dim=1)
            spread = raw_energy.max(dim=1).values - raw_energy.min(dim=1).values

        peak = probabilities.max(dim=1).values

        return {

            "energy": energy,

            "raw_energy": raw_energy,

            "probabilities": probabilities,

            "amplitude": amplitude,

            "temperature": temperature.mean().detach(),

            "confidence": confidence,

            "entropy": entropy.mean(),

            "diversity": diversity.mean(),

            "spread": spread.mean(),

            "peak": peak.mean(),

        }

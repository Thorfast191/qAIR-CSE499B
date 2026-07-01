import torch
import torch.nn as nn
import torch.nn.functional as F


class CollapseController(nn.Module):
    """
    Adaptive Quantum Collapse Controller

    Improvements
    ------------
    • Per-sample temperature
    • Per-sample collapse bias
    • Confidence estimation
    • Numerically stable Born Rule
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
        # Adaptive bias
        ####################################################

        self.bias = nn.Linear(32, 1)

        ####################################################
        # Confidence
        ####################################################

        self.confidence = nn.Sequential(
            nn.Linear(32,16),
            nn.GELU(),
            nn.Linear(16,1),
            nn.Sigmoid(),
        )

    def forward(self, energy):

        raw_energy = energy

        ####################################################
        # Normalize
        ####################################################

        energy = energy - energy.mean(
            dim=1,
            keepdim=True,
        )

        energy = energy / (
            energy.std(
                dim=1,
                keepdim=True,
            )
            + 1e-6
        )

        ####################################################
        # Encode energy distribution
        ####################################################

        z = self.encoder(
            energy.unsqueeze(-1)
        )

        z_global = z.mean(dim=1)

        ####################################################
        # Adaptive parameters
        ####################################################

        temperature = (
            0.5
            +
            F.softplus(
                self.temperature(z_global)
            )
        )

        bias = self.bias(z_global)

        confidence = self.confidence(z_global).squeeze(-1)

        ####################################################
        # Stable Born Rule
        ####################################################

        log_amp = -energy / temperature + bias

        log_amp = log_amp - log_amp.max(
            dim=1,
            keepdim=True,
        ).values

        amplitude = torch.exp(log_amp)

        amplitude = amplitude / (
            amplitude.sum(
                dim=1,
                keepdim=True,
            )
            + 1e-8
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
        # Metrics
        ####################################################

        entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=1)

        diversity = raw_energy.var(dim=1)

        spread = raw_energy.max(dim=1).values - raw_energy.min(dim=1).values

        peak = probabilities.max(dim=1).values

        ####################################################
        # Adaptive collapse objective
        ####################################################

        target_entropy = 1.0 - confidence

        collapse_loss = (

            0.10 * (entropy - target_entropy).pow(2).mean()

            +

            0.01 * diversity.mean()

            +

            0.005 * spread.mean()

        )

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

            "collapse_loss": collapse_loss,

        }
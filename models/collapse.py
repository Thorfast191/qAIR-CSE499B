import torch
import torch.nn as nn
import torch.nn.functional as F


class CollapseController(nn.Module):
    """
    Learnable Quantum Collapse Controller

    Input
    -----
    energy : (B, K)

    Returns
    -------
    probabilities : (B, K)

    Notes
    -----
    - Lower energy -> higher collapse probability
    - Uses Boltzmann amplitudes followed by the Born Rule
    """

    def __init__(self):

        super().__init__()

        # Learnable temperature
        # Effective temperature:
        # T = 0.5 + softplus(parameter)
        self.temperature = nn.Parameter(torch.tensor(0.0))

        # Learnable bias
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, energy):

        # =====================================================
        # Preserve raw energy for metrics
        # =====================================================

        raw_energy = energy.clone()

        # =====================================================
        # Normalize energy
        # =====================================================

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

        energy = torch.clamp(
            energy,
            -3.0,
            3.0,
        )

        # =====================================================
        # Boltzmann Amplitudes
        # =====================================================

        temperature = 0.5 + F.softplus(self.temperature)

        amplitude = torch.exp(-energy / temperature + self.bias)

        # Normalize amplitudes

        amplitude = amplitude / (
            amplitude.sum(
                dim=1,
                keepdim=True,
            )
            + 1e-8
        )

        # =====================================================
        # Born Rule
        # =====================================================

        probabilities = amplitude.pow(2)

        probabilities = probabilities / (
            probabilities.sum(
                dim=1,
                keepdim=True,
            )
            + 1e-8
        )

        # =====================================================
        # Metrics
        # =====================================================

        entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=1).mean()

        diversity = raw_energy.var(dim=1).mean()

        spread = (raw_energy.max(dim=1).values - raw_energy.min(dim=1).values).mean()

        peak = probabilities.max(dim=1).values.mean()

        # =====================================================
        # Stable Collapse Loss
        # =====================================================

        target_diversity = 1.0
        target_spread = 2.0

        collapse_loss = (
            0.05 * entropy
            + 0.01 * (diversity - target_diversity).pow(2)
            + 0.005 * (spread - target_spread).pow(2)
        )

        return {
            "energy": energy,
            "raw_energy": raw_energy,
            "temperature": temperature.detach(),
            "amplitude": amplitude,
            "probabilities": probabilities,
            "entropy": entropy,
            "diversity": diversity,
            "spread": spread,
            "peak": peak,
            "collapse_loss": collapse_loss,
        }

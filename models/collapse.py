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
    """

    def __init__(self):

        super().__init__()

        # Learnable inverse temperature
        self.temperature = nn.Parameter(
            torch.tensor(2.0)
        )

        # Learnable collapse bias
        self.bias = nn.Parameter(
            torch.zeros(1)
        )

    def forward(self, energy):

        # ---------------------------------------------
        # Normalize energy (stable optimization)
        # ---------------------------------------------

        energy = energy - energy.mean(
            dim=1,
            keepdim=True,
        )

        energy = energy / (
            energy.std(
                dim=1,
                keepdim=True,
            ) + 1e-6
        )

        # ---------------------------------------------
        # Quantum amplitudes
        # ---------------------------------------------

        amplitude = torch.exp(
            -self.temperature * energy + self.bias
        )

        amplitude = amplitude / (
            amplitude.sum(
                dim=1,
                keepdim=True,
            ) + 1e-8
        )

        # Born Rule
        probabilities = amplitude.pow(2)

        probabilities = probabilities / (
            probabilities.sum(
                dim=1,
                keepdim=True,
            ) + 1e-8
        )

        # ---------------------------------------------
        # Metrics
        # ---------------------------------------------

        entropy = -(
            probabilities *
            torch.log(
                probabilities + 1e-8
            )
        ).sum(dim=1).mean()

        diversity = energy.var(
            dim=1
        ).mean()

        spread = (
            energy.max(dim=1).values
            -
            energy.min(dim=1).values
        ).mean()

        peak = probabilities.max(
            dim=1
        ).values.mean()

        # ---------------------------------------------
        # Collapse loss
        # ---------------------------------------------

        collapse_loss = (
            0.05 * entropy
            - 0.05 * diversity
            - 0.02 * spread
        )

        return {

            "energy": energy,

            "amplitude": amplitude,

            "probabilities": probabilities,

            "entropy": entropy,

            "diversity": diversity,

            "spread": spread,

            "peak": peak,

            "collapse_loss": collapse_loss,
        }
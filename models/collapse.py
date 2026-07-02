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

        energy = torch.tanh(energy)

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
            2.5
            +
            F.softplus(
                self.temperature(z_global)
            )
        )

        confidence = self.confidence(z_global).squeeze(-1)

        ####################################################
        # Stable Born Rule
        ####################################################

        log_amp = -energy / (2.0 * temperature) 

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
        # Metrics
        ####################################################

        entropy = -(probabilities * torch.log(probabilities + 1e-8)).sum(dim=1)

        diversity = raw_energy.var(dim=1)

        spread = raw_energy.max(dim=1).values - raw_energy.min(dim=1).values

        peak = probabilities.max(dim=1).values

        ####################################################
        # Adaptive collapse objective
        ####################################################

        target_entropy = torch.full_like(entropy, 1.2)

        collapse_loss = (

            0.10 * (entropy - target_entropy).pow(2).mean()

            +

            0.01 * diversity.mean()

            +

            0.005 * spread.mean()

        )

        if self.training and torch.rand(1).item() < 0.01:
            print(f"Temp: {temperature.mean().item():.3f}")
            print(f"Entropy: {entropy.mean().item():.3f}")
            print(f"Peak: {peak.mean().item():.3f}")
            print(f"Energy std: {raw_energy.std().item():.3f}")

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
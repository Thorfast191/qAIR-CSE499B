import torch
import torch.nn as nn


class EnergyFusion(nn.Module):
    """
    Combines the answer selector's per-hypothesis energy with the optional
    quantum and validator energy signals into a single collapse energy,
    via learned sigmoid-gated blend weights.
    """

    def __init__(self):

        super().__init__()

        self.energy_alpha = nn.Parameter(torch.tensor(0.0))

        # Learnable validator guidance weight. sigmoid(0.0) = 0.5 at init --
        # moderate influence, model can learn to trust it more or suppress
        # it based on whether the validator's signal actually helps.
        self.validator_alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, answer_energy, quantum_energy=None, validator_energy=None, tau=1.0):

        collapse_energy = torch.min(answer_energy / tau, dim=-1).values

        if quantum_energy is not None:

            alpha = torch.sigmoid(self.energy_alpha)

            collapse_energy = alpha * collapse_energy + (1.0 - alpha) * quantum_energy

        if validator_energy is not None:

            v_weight = torch.sigmoid(self.validator_alpha)

            collapse_energy = collapse_energy + v_weight * validator_energy

        return collapse_energy

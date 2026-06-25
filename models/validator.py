import torch
import torch.nn as nn
import torch.nn.functional as F


class HypothesisValidator(nn.Module):
    """
    Quantum-Inspired Hypothesis Validator

    Four observable energies:

        • causal
        • diversity
        • specificity
        • relevance

    These are combined into a Hamiltonian energy,
    converted into Born probabilities,
    and used to generate a reasoning potential.

    No Softmax.
    """

    def __init__(self, dim):

        super().__init__()

        self.score_net = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.LayerNorm(dim * 2),
            nn.Dropout(0.10),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )

        self.potential_net = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

        # Learnable observable coefficients
        self.observable_weights = nn.Parameter(
            torch.tensor(
                [
                    0.40,
                    0.20,
                    0.20,
                    0.20,
                ]
            )
        )

        # Learnable temperature
        self.temperature = nn.Parameter(
            torch.tensor(1.0)
        )

    def forward(self, H, O, y=None):

        B, K, D = H.shape
        _, N, _ = O.shape

        H = F.normalize(H, dim=-1)
        O = F.normalize(O, dim=-1)

        H_exp = H.unsqueeze(2).expand(
            B, K, N, D
        )

        O_exp = O.unsqueeze(1).expand(
            B, K, N, D
        )

        diff = H_exp - O_exp
        prod = H_exp * O_exp

        features = torch.cat(
            [
                H_exp,
                O_exp,
                diff,
                prod,
            ],
            dim=-1,
        )

        scores = self.score_net(features)

        causal = scores[..., 0]
        diversity = scores[..., 1]
        specificity = scores[..., 2]
        relevance = scores[..., 3]

        target = None

        if y is not None:

            target = torch.zeros_like(relevance)

            target[
                torch.arange(B, device=H.device),
                :,
                y,
            ] = 1.0

        ####################################################
        # Potential
        ####################################################

        potential = self.potential_net(features)

        ####################################################
        # Hamiltonian Energy
        ####################################################

        w = F.softplus(
            self.observable_weights
        )

        w = w / (
            w.sum() + 1e-8
        )

        quality_energy = (

            w[0] * causal

            +

            w[1] * diversity

            +

            w[2] * specificity

            +

            w[3] * relevance

        )

        ####################################################
        # One energy per hypothesis
        ####################################################

        quality_energy = quality_energy.mean(
            dim=2
        )

        ####################################################
        # Born Rule
        ####################################################

        energy = -quality_energy

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
            -3,
            3,
        )

        temperature = (
            0.5
            +
            F.softplus(
                self.temperature
            )
        )

        amplitude = torch.exp(
            -energy / temperature
        )

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
        # Weight potential
        ####################################################

        potential = potential.mean(dim=2)

        potential = (
            potential
            *
            probabilities.unsqueeze(-1)
        )

        return {

            "potential": potential,

            "validator_energy": quality_energy,

            "validator_probabilities": probabilities,

            "causal": causal.mean(dim=2),

            "diversity": diversity.mean(dim=2),

            "specificity": specificity.mean(dim=2),

            "relevance": relevance.mean(dim=2),

            "relevance_logits": relevance,

            "relevance_target": target,

        }
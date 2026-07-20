import torch
import torch.nn as nn
import torch.nn.functional as F


class HypothesisValidator(nn.Module):
    """
    Adaptive Quantum Hypothesis Validator

    Improvements
    ------------
    • Adaptive observable weighting
    • Reliability estimation
    • Stable Born collapse
    • Confidence-aware reasoning potential
    """

    def __init__(self, dim):

        super().__init__()

        ####################################################
        # Shared feature encoder
        ####################################################

        self.encoder = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.LayerNorm(dim * 2),
            nn.Dropout(0.10),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
        )

        ####################################################
        # Observable heads
        ####################################################

        self.causal = nn.Linear(dim, 1)
        self.diversity = nn.Linear(dim, 1)
        self.specificity = nn.Linear(dim, 1)
        self.relevance = nn.Linear(dim, 1)

        ####################################################
        # NEW
        # Adaptive observable fusion
        ####################################################

        self.observable_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )

        ####################################################
        # Reliability prediction
        ####################################################

        self.reliability = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

        ####################################################
        # Reasoning potential
        ####################################################

        self.potential = nn.Sequential(
            nn.Linear(dim + 1, dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(dim, dim),
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, H, O, y=None):

        B, K, D = H.shape
        _, N, _ = O.shape

        H = F.normalize(H, dim=-1)
        O = F.normalize(O, dim=-1)

        H_exp = H.unsqueeze(2).expand(B, K, N, D)
        O_exp = O.unsqueeze(1).expand(B, K, N, D)

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

        ####################################################
        # Shared representation
        ####################################################

        z = self.encoder(features)

        ####################################################
        # Observables
        ####################################################

        causal = self.causal(z).squeeze(-1)
        diversity = self.diversity(z).squeeze(-1)
        specificity = self.specificity(z).squeeze(-1)
        relevance = self.relevance(z).squeeze(-1)

        ####################################################
        # Supervised relevance
        ####################################################

        target = None

        if y is not None:

            target = torch.zeros_like(relevance)

            target[
                torch.arange(B, device=H.device),
                :,
                y,
            ] = 1.0

        ####################################################
        # Adaptive observable weights
        ####################################################

        z_mean = z.mean(dim=2)

        gate = self.observable_gate(
            z_mean
        )

        weights = F.softmax(
            gate,
            dim=-1,
        )

        energy = (

            weights[..., 0] * causal.mean(dim=2)

            +

            weights[..., 1] * diversity.mean(dim=2)

            +

            weights[..., 2] * specificity.mean(dim=2)

            +

            weights[..., 3] * relevance.mean(dim=2)

        )

        ####################################################
        # Stable Born Rule
        ####################################################

        energy = -energy

        temperature = (
            0.5
            + F.softplus(self.temperature)
        )

        log_amp = -energy / temperature

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
        # Reliability
        ####################################################

        reliability = self.reliability(
            z_mean
        )

        reliability = reliability.squeeze(-1)

        ####################################################
        # Confidence-aware potential
        ####################################################

        potential_input = torch.cat(
            [
                z_mean,
                reliability.unsqueeze(-1),
            ],
            dim=-1,
        )

        potential = self.potential(
            potential_input
        )

        potential = (
            potential
            *
            probabilities.unsqueeze(-1)
        )

        return {

            "potential": potential,

            "validator_energy": energy,

            "validator_probabilities": probabilities,

            "reliability": reliability,

            "observable_weights": weights,

            "causal": causal.mean(dim=2),

            "diversity": diversity.mean(dim=2),

            "specificity": specificity.mean(dim=2),

            "relevance": relevance.mean(dim=2),

            "relevance_logits": relevance,

            "relevance_target": target,

        }
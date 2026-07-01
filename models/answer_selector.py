import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyAnswerSelector(nn.Module):
    """
    Adaptive Hamiltonian Energy Selector

    Lower Energy = Better Answer

    Improvements
    ------------
    • Dynamic energy fusion
    • Adaptive Hamiltonian metric
    • Confidence estimation
    • Stable energy normalization
    """

    def __init__(self, dim):

        super().__init__()

        # ------------------------------------------
        # Hamiltonian projection
        # ------------------------------------------

        self.hamiltonian = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        # ------------------------------------------
        # Learned compatibility
        # ------------------------------------------

        self.energy_net = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.LayerNorm(dim * 2),
            nn.Dropout(0.10),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

        # ------------------------------------------
        # NEW
        # Adaptive fusion network
        # ------------------------------------------

        self.fusion = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
        )

        # ------------------------------------------
        # NEW
        # Confidence estimation
        # ------------------------------------------

        self.confidence = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, H, O):

        B, K, D = H.shape
        _, N, _ = O.shape

        ##################################################
        # Normalize
        ##################################################

        H = F.normalize(H, dim=-1)
        O = F.normalize(O, dim=-1)

        O_proj = self.hamiltonian(O)

        ##################################################
        # Pairwise tensors
        ##################################################

        H_exp = H.unsqueeze(2).expand(B, K, N, D)
        O_exp = O_proj.unsqueeze(1).expand(B, K, N, D)

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

        ##################################################
        # Learned Energy
        ##################################################

        learned_energy = self.energy_net(features).squeeze(-1)

        ##################################################
        # Hamiltonian Energy
        ##################################################

        hamiltonian_energy = -torch.einsum(
            "bkd,bnd->bkn",
            H,
            O_proj,
        )

        ##################################################
        # Cosine Energy
        ##################################################

        cosine_energy = -F.cosine_similarity(
            H_exp,
            O_exp,
            dim=-1,
        )

        ##################################################
        # Adaptive Fusion
        ##################################################

        fusion_logits = self.fusion(features)

        fusion_weights = F.softmax(
            fusion_logits,
            dim=-1,
        )

        energy = (

            fusion_weights[..., 0] * learned_energy

            +

            fusion_weights[..., 1] * hamiltonian_energy

            +

            fusion_weights[..., 2] * cosine_energy

        )

        ##################################################
        # Confidence
        ##################################################

        confidence = self.confidence(features).squeeze(-1)

        energy = energy / confidence.clamp(min=0.2)

        ##################################################
        # Stable normalization
        ##################################################

        energy = energy - energy.mean(
            dim=-1,
            keepdim=True,
        )

        energy = energy / (
            energy.std(
                dim=-1,
                keepdim=True,
            )
            + 1e-6
        )

        ##################################################
        # Temperature
        ##################################################

        temperature = 0.5 + F.softplus(
            self.temperature
        )

        energy = energy / temperature

        energy = torch.clamp(
            energy,
            -6,
            6,
        )

        return {
            "energy": energy,
            "confidence": confidence,
            "fusion_weights": fusion_weights,
        }
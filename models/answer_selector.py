import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyAnswerSelector(nn.Module):
    """
    Hamiltonian Energy Network

    Inputs
    ------
    H : (B, K, D)
        Hypothesis embeddings

    O : (B, N, D)
        Answer embeddings

    Returns
    -------
    energy : (B, K, N)

    Lower energy = better hypothesis-answer compatibility
    """

    def __init__(self, dim):

        super().__init__()

        # Learnable Hamiltonian
        self.hamiltonian = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        # Neural energy function
        self.energy_net = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.LayerNorm(dim * 2),
            nn.Dropout(0.10),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

        # Learnable positive temperature
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, H, O):

        B, K, D = H.shape
        _, N, _ = O.shape

        # ===================================================
        # Normalize representations
        # ===================================================

        H = F.normalize(
            H,
            dim=-1,
        )

        O = F.normalize(
            O,
            dim=-1,
        )

        # ===================================================
        # Hamiltonian projection
        # ===================================================

        O_proj = self.hamiltonian(O)

        # ===================================================
        # Pairwise expansion
        # ===================================================

        H_exp = H.unsqueeze(2).expand(
            B,
            K,
            N,
            D,
        )

        O_exp = O_proj.unsqueeze(1).expand(
            B,
            K,
            N,
            D,
        )

        # ===================================================
        # Interaction Features
        # ===================================================

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

        # ===================================================
        # Neural Hamiltonian Energy
        # ===================================================

        learned_energy = self.energy_net(features).squeeze(-1)

        # ===================================================
        # Hamiltonian Expectation
        # ===================================================

        hamiltonian_energy = -torch.einsum(
            "bkd,bnd->bkn",
            H,
            O_proj,
        )

        # ===================================================
        # Cosine Similarity Prior
        # ===================================================

        cosine_energy = -F.cosine_similarity(
            H_exp,
            O_exp,
            dim=-1,
        )

        # ===================================================
        # Final Energy
        # ===================================================

        energy = learned_energy + 0.5 * hamiltonian_energy + 0.5 * cosine_energy

        # ===================================================
        # Positive Temperature
        # ===================================================

        temperature = F.softplus(self.temperature)

        energy = energy * temperature

        return energy

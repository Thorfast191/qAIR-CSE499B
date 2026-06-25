import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyAnswerSelector(nn.Module):
    """
    Stable Hamiltonian Energy Network

    Lower energy = better hypothesis-answer compatibility
    """

    def __init__(self, dim):

        super().__init__()

        self.hamiltonian = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.energy_net = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.LayerNorm(dim * 2),
            nn.Dropout(0.10),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

        # learnable positive temperature

        self.temperature = nn.Parameter(torch.tensor(1.0))

        # learnable weighting

        self.learned_weight = nn.Parameter(torch.tensor(1.0))

        self.hamiltonian_weight = nn.Parameter(torch.tensor(0.5))

        self.cosine_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, H, O):

        B, K, D = H.shape
        _, N, _ = O.shape

        # --------------------------------------------------
        # Normalize embeddings
        # --------------------------------------------------

        H = F.normalize(H, dim=-1)

        O = F.normalize(O, dim=-1)

        O_proj = self.hamiltonian(O)

        # --------------------------------------------------
        # Pairwise tensors
        # --------------------------------------------------

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

        # --------------------------------------------------
        # Learned energy
        # --------------------------------------------------

        learned_energy = self.energy_net(features).squeeze(-1)

        # Prevent runaway values

        learned_energy = 3.0 * torch.tanh(learned_energy / 3.0)

        # --------------------------------------------------
        # Hamiltonian expectation
        # --------------------------------------------------

        hamiltonian_energy = -torch.einsum(
            "bkd,bnd->bkn",
            H,
            O_proj,
        )

        # Already naturally bounded approximately [-1,1]

        hamiltonian_energy = torch.clamp(
            hamiltonian_energy,
            -2.0,
            2.0,
        )

        # --------------------------------------------------
        # Cosine energy
        # --------------------------------------------------

        cosine_energy = -F.cosine_similarity(
            H_exp,
            O_exp,
            dim=-1,
        )

        # --------------------------------------------------
        # Learnable fusion
        # --------------------------------------------------

        lw = torch.sigmoid(self.learned_weight)

        hw = torch.sigmoid(self.hamiltonian_weight)

        cw = torch.sigmoid(self.cosine_weight)

        energy = lw * learned_energy + hw * hamiltonian_energy + cw * cosine_energy

        # --------------------------------------------------
        # Stable temperature
        # --------------------------------------------------

        temperature = 0.5 + F.softplus(self.temperature)

        energy = energy / temperature

        # Final safeguard

        energy = torch.clamp(
            energy,
            -5.0,
            5.0,
        )

        return energy

import torch
import torch.nn as nn


class PersistentReasoner(nn.Module):

    def __init__(self, dim, steps=3):

        super().__init__()

        self.steps = steps

        self.hamiltonian = nn.Linear(dim, dim, bias=False)

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim)
        )

        self.norm1 = nn.LayerNorm(dim)

        self.norm2 = nn.LayerNorm(dim)

        self.dt = nn.Parameter(torch.tensor(0.1))

    def forward(self, H):

        trajectory = []

        for _ in range(self.steps):

            # pairwise hypothesis interaction

            interaction = torch.einsum("bkd,bjd->bkj", H, H)

            interaction = interaction / (H.size(-1) ** 0.5)

            # Hamiltonian field

            field = torch.einsum("bkj,bjd->bkd", interaction, H)

            field = self.hamiltonian(field)

            # quantum-style evolution

            H = self.norm1(H + self.dt * field)

            H = self.norm2(H + self.ffn(H))

            trajectory.append(H.detach().cpu())

        return H, trajectory, interaction

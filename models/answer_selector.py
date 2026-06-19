import torch
import torch.nn as nn


class EnergyAnswerSelector(nn.Module):

    def __init__(self, dim):

        super().__init__()

        # Learnable Hamiltonian operator
        self.hamiltonian = nn.Linear(
            dim,
            dim,
            bias=False
        )

    def forward(self, H, O):

        """
        H : (B, K, D)
            K hypothesis states

        O : (B, N, D)
            N answer option states

        Returns
        -------
        energy : (B, K, N)

        Lower energy => better answer
        """

        # Apply Hamiltonian to answer states
        O_proj = self.hamiltonian(O)

        # Hamiltonian expectation
        energy = -torch.einsum(
            "bkd,bnd->bkn",
            H,
            O_proj
        )

        return energy
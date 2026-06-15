import torch
import torch.nn as nn


class EnergyAnswerSelector(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1)
        )

    def forward(self, H, O):

        B, K, D = H.shape
        _, N, _ = O.shape

        H_exp = H.unsqueeze(2).expand(
            B, K, N, D
        )

        O_exp = O.unsqueeze(1).expand(
            B, K, N, D
        )

        x = torch.cat(
            [H_exp, O_exp],
            dim=-1
        )

        energy = self.net(x).squeeze(-1)

        return energy
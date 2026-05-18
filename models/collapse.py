import torch
import torch.nn as nn


class CollapseController(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.energy_head = nn.Linear(dim, 1)

    def forward(self, H):

        E = self.energy_head(H).squeeze(-1)

        spread = E.var(dim=1).mean()

        collapse_loss = -spread

        return E, collapse_loss
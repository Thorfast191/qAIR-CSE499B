import torch
import torch.nn as nn


class PersistentReasoner(nn.Module):

    def __init__(self, dim, steps=3):
        super().__init__()

        self.steps = steps

        self.attn = nn.MultiheadAttention(
            dim,
            4,
            batch_first=True
        )

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, H):

        trajectory = []

        for _ in range(self.steps):

            attn_out, attn_weights = self.attn(H, H, H)

            H = self.norm1(H + attn_out)
            H = self.norm2(H + self.ffn(H))

            trajectory.append(H.detach().cpu())

        return H, trajectory, attn_weights
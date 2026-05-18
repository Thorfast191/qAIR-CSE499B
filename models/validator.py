import torch
import torch.nn as nn
import torch.nn.functional as F


class HypothesisValidator(nn.Module):

    def __init__(self, dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim, 4)
        )

    def forward(self, H, O):

        B, K, D = H.shape

        Omean = O.mean(dim=1).unsqueeze(1).expand(B, K, D)

        x = torch.cat([H, Omean], dim=-1)

        scores = self.net(x)

        causal = scores[..., 0]
        diversity = scores[..., 1]
        specificity = scores[..., 2]
        relevance = scores[..., 3]

        energy = -(
            0.4 * causal +
            0.2 * diversity +
            0.2 * specificity +
            0.2 * relevance
        )

        return {
            "energy": energy,
            "causal": causal,
            "diversity": diversity,
            "specificity": specificity,
            "relevance": relevance
        }
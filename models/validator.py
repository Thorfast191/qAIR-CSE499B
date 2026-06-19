import torch
import torch.nn as nn


class HypothesisValidator(nn.Module):

    def __init__(self, dim):

        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(dim * 2, dim), nn.GELU(), nn.Dropout(0.1), nn.Linear(dim, 4)
        )

    def forward(self, H, O):

        B, K, D = H.shape
        _, N, _ = O.shape

        H_exp = H.unsqueeze(2).expand(B, K, N, D)

        O_exp = O.unsqueeze(1).expand(B, K, N, D)

        x = torch.cat([H_exp, O_exp], dim=-1)

        scores = self.net(x)

        causal = scores[..., 0]
        diversity = scores[..., 1]
        specificity = scores[..., 2]
        relevance = scores[..., 3]

        potential = -(0.4 * causal + 0.2 * diversity + 0.2 * specificity + 0.2 * relevance)

        potential = potential.mean(dim=2)

        return {
            "potential": potential,
            "causal": causal.mean(dim=2),
            "diversity": diversity.mean(dim=2),
            "specificity": specificity.mean(dim=2),
            "relevance": relevance.mean(dim=2),
        }

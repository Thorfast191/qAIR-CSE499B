import torch
import torch.nn as nn
import torch.nn.functional as F


class HypothesisValidator(nn.Module):
    """
    Multi-objective Hypothesis Validator

    Evaluates every hypothesis against every option and produces

    • causal score
    • diversity score
    • specificity score
    • relevance score

    and a learned vector potential used by the
    PersistentReasoner.
    """

    def __init__(self, dim):

        super().__init__()

        self.score_net = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.LayerNorm(dim * 2),
            nn.Dropout(0.10),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )

        # Feature-wise Hamiltonian potential
        self.potential_net = nn.Sequential(
            nn.Linear(dim * 4, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, H, O, y=None):

        B, K, D = H.shape
        _, N, _ = O.shape

        # Normalize embeddings

        H = F.normalize(H, dim=-1)

        O = F.normalize(O, dim=-1)

        H_exp = H.unsqueeze(2).expand(
            B,
            K,
            N,
            D,
        )

        O_exp = O.unsqueeze(1).expand(
            B,
            K,
            N,
            D,
        )

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

        # -----------------------------------------
        # Validation scores
        # -----------------------------------------

        scores = self.score_net(features)

        causal = scores[..., 0]

        diversity = scores[..., 1]

        specificity = scores[..., 2]

        relevance = scores[..., 3]

        target = None

        if y is not None:
            target = torch.zeros_like(relevance)
            target[
                torch.arange(B, device=H.device),
                :,
                y,
            ] = 1.0
        # -----------------------------------------
        # Feature-wise potential
        # -----------------------------------------

        potential = self.potential_net(features)

        # Weight by reasoning quality

        weights = torch.softmax(
            torch.stack(
                [
                    causal,
                    diversity,
                    specificity,
                    relevance,
                ],
                dim=-1,
            ),
            dim=-1,
        )

        quality = (
            0.40 * weights[..., 0]
            + 0.20 * weights[..., 1]
            + 0.20 * weights[..., 2]
            + 0.20 * weights[..., 3]
        )

        potential = potential * quality.unsqueeze(-1)

        # Average over answer options

        potential = potential.mean(dim=2)

        return {
            "potential": potential,
            "causal": causal.mean(dim=2),
            "diversity": diversity.mean(dim=2),
            "specificity": specificity.mean(dim=2),
            "relevance": relevance.mean(dim=2),
            "relevance_logits": relevance,
            "relevance_target": target,
        }

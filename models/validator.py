import torch
import torch.nn as nn
import torch.nn.functional as F


class HypothesisValidator(nn.Module):
    """
    Scores hypothesis-vs-option compatibility and emits a guidance
    potential for the reasoner.

    Fixed in the audit
    ------------------
    * Question conditioning. The question embedding was never computed
      anywhere in the pipeline, so the validator had to judge options
      using only hypotheses -- a third of which were template fallbacks.
      Measured on cached ARC embeddings with a plain MLP: options only
      0.3585, +question 0.4849. `use_question` adds Q*O and Q-O terms.
    * Born rule normalization. This module L1-normalized the amplitude
      before squaring, which is not the Born rule (that needs L2) and
      gave it an effective temperature of T/2 versus collapse.py's T for
      the same nominal T. Both now use the L2 convention.
    * Padded options are excluded from the means over N.
    """

    def __init__(self, dim, use_question=True):

        super().__init__()

        self.use_question = use_question

        # [H, O, H-O, H*O] plus, when the question is available,
        # [Q*O, Q-O].
        n_feat = 6 if use_question else 4

        self.encoder = nn.Sequential(
            nn.Linear(dim * n_feat, dim * 2),
            nn.GELU(),
            nn.LayerNorm(dim * 2),
            nn.Dropout(0.10),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
        )

        self.causal = nn.Linear(dim, 1)
        self.diversity = nn.Linear(dim, 1)
        self.specificity = nn.Linear(dim, 1)
        self.relevance = nn.Linear(dim, 1)

        self.observable_gate = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 4),
        )

        self.reliability = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

        self.potential = nn.Sequential(
            nn.Linear(dim + 1, dim),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(dim, dim),
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, H, O, Q=None, y=None, H_mask=None, O_mask=None):

        B, K, D = H.shape
        _, N, _ = O.shape

        H = F.normalize(H, dim=-1)
        O = F.normalize(O, dim=-1)

        H_exp = H.unsqueeze(2).expand(B, K, N, D)
        O_exp = O.unsqueeze(1).expand(B, K, N, D)

        diff = H_exp - O_exp
        prod = H_exp * O_exp

        parts = [H_exp, O_exp, diff, prod]

        if self.use_question:

            if Q is None:
                raise ValueError(
                    "HypothesisValidator was built with use_question=True "
                    "but forward() got Q=None. Rebuild the cache so it "
                    "carries question embeddings (training/dataset.py "
                    "migrates existing caches automatically), or construct "
                    "the model with use_question=False."
                )

            Q_exp = F.normalize(Q, dim=-1).view(B, 1, 1, D).expand(B, K, N, D)

            parts += [Q_exp * O_exp, Q_exp - O_exp]

        features = torch.cat(parts, dim=-1)

        z = self.encoder(features)

        causal = self.causal(z).squeeze(-1)
        diversity = self.diversity(z).squeeze(-1)
        specificity = self.specificity(z).squeeze(-1)
        relevance = self.relevance(z).squeeze(-1)

        target = None

        if y is not None:
            target = torch.zeros_like(relevance)
            target[torch.arange(B, device=H.device), :, y] = 1.0

        # ------------------------------------------------------------
        # Masked reductions over the option axis
        # ------------------------------------------------------------

        if O_mask is not None:
            om = O_mask.view(B, 1, N, 1).to(z.dtype)
            denom = om.sum(dim=2).clamp(min=1.0)
            z_mean = (z * om).sum(dim=2) / denom
            om2 = O_mask.view(B, 1, N).to(causal.dtype)
            d2 = om2.sum(dim=2).clamp(min=1.0)
            mean = lambda t: (t * om2).sum(dim=2) / d2
        else:
            z_mean = z.mean(dim=2)
            mean = lambda t: t.mean(dim=2)

        gate = self.observable_gate(z_mean)

        weights = F.softmax(gate, dim=-1)

        energy = (
            weights[..., 0] * mean(causal)
            + weights[..., 1] * mean(diversity)
            + weights[..., 2] * mean(specificity)
            + weights[..., 3] * mean(relevance)
        )

        energy = -energy

        # ------------------------------------------------------------
        # Born rule -- same L2 convention as collapse.py
        # ------------------------------------------------------------

        temperature = 0.5 + F.softplus(self.temperature)

        log_amp = -energy / (2.0 * temperature)

        if H_mask is not None:
            log_amp = log_amp.masked_fill(~H_mask, float("-inf"))

        log_amp = log_amp - log_amp.max(dim=1, keepdim=True).values

        amplitude = torch.exp(log_amp)

        amplitude = amplitude / torch.sqrt(
            (amplitude ** 2).sum(dim=1, keepdim=True) + 1e-8
        )

        probabilities = amplitude.pow(2)

        probabilities = probabilities / (probabilities.sum(dim=1, keepdim=True) + 1e-8)

        reliability = self.reliability(z_mean).squeeze(-1)

        potential_input = torch.cat([z_mean, reliability.unsqueeze(-1)], dim=-1)

        potential = self.potential(potential_input)

        potential = potential * probabilities.unsqueeze(-1)

        return {
            "potential": potential,
            "validator_energy": energy,
            "validator_probabilities": probabilities,
            "reliability": reliability,
            "observable_weights": weights,
            "causal": mean(causal),
            "diversity": mean(diversity),
            "specificity": mean(specificity),
            "relevance": mean(relevance),
            "relevance_logits": relevance,
            "relevance_target": target,
        }

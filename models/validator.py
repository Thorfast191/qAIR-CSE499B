import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DROPOUT


class HypothesisValidator(nn.Module):
    """
    Scores hypothesis-vs-option compatibility and emits a guidance
    potential for the reasoner.

    On the "Born rule" here
    -----------------------
    The distribution computed below weights the guidance potential. With
    real, non-negative amplitudes it is a temperature-scaled softmax and
    nothing more -- that is stated rather than dressed up. The place
    where amplitudes are genuinely complex, and where the Born rule
    therefore does real work, is models/interference.py; this module
    feeds it (through `validator_energy`) but does not itself perform a
    measurement.

    Fixed in the v44 audit, kept
    ----------------------------
    * Question conditioning. The question embedding was never computed
      anywhere in the pipeline, so the validator judged options using
      hypotheses alone -- a third of which were template fallbacks.
      Measured with a plain MLP on cached ARC embeddings: options only
      0.3585, +question 0.4849.
    * L2 (not L1) normalization of the amplitude. L1 is not the Born
      convention and gave this module half the collapse controller's
      effective temperature at the same nominal T.
    * Padded options are excluded from every mean over N.

    New in v45
    ----------
    * `align` / `polarity` features, so the validator can distinguish
      "the support for option n" from "the objection to option n"
      instead of averaging them together.
    """

    def __init__(self, dim, use_question=True):

        super().__init__()

        self.use_question = use_question

        # [H, O, H-O, H*O] plus, when the question is available,
        # [Q*O, Q-O]; then three scalars: align, polarity, align*polarity.
        n_vec = 6 if use_question else 4

        n_feat = dim * n_vec + 3

        self.encoder = nn.Sequential(
            nn.Linear(n_feat, dim * 2),
            nn.GELU(),
            nn.LayerNorm(dim * 2),
            nn.Dropout(DROPOUT),
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
            nn.Dropout(DROPOUT),
            nn.Linear(dim, dim),
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, H, O, Q=None, y=None, H_mask=None, O_mask=None,
                align=None, polarity=None):

        B, K, D = H.shape
        _, N, _ = O.shape

        H = F.normalize(H, dim=-1)
        O = F.normalize(O, dim=-1)

        H_exp = H.unsqueeze(2).expand(B, K, N, D)
        O_exp = O.unsqueeze(1).expand(B, K, N, D)

        parts = [H_exp, O_exp, H_exp - O_exp, H_exp * O_exp]

        if self.use_question:

            if Q is None:
                raise ValueError(
                    "HypothesisValidator was built with use_question=True "
                    "but forward() got Q=None. Rebuild the cache so it "
                    "carries question embeddings, or construct the model "
                    "with use_question=False."
                )

            Q_exp = F.normalize(Q, dim=-1).view(B, 1, 1, D).expand(B, K, N, D)

            parts += [Q_exp * O_exp, Q_exp - O_exp]

        if align is None:
            align = torch.zeros(B, K, N, device=H.device, dtype=H.dtype)

        if polarity is None:
            polarity = torch.zeros(B, K, device=H.device, dtype=H.dtype)

        a = align.unsqueeze(-1)
        p = polarity.view(B, K, 1, 1).expand(B, K, N, 1)

        parts += [a, p, a * p]

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
        # Weighting distribution over hypotheses (see class docstring:
        # this is a softmax, and is not claimed to be more)
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

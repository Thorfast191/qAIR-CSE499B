import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DROPOUT


def bounded_phase(x):
    """Bound an unconstrained head into (-pi, pi]."""

    return math.pi * torch.tanh(x)


def _masked_standardize(x, valid, eps=1e-5):
    """
    Zero-mean, unit-variance per sample over the (K, N) axes, ignoring
    padded entries. Padded entries are returned as 0.
    """

    m = valid.to(x.dtype)

    n = m.sum(dim=(1, 2), keepdim=True).clamp(min=1.0)

    mean = (x * m).sum(dim=(1, 2), keepdim=True) / n

    var = (((x - mean) ** 2) * m).sum(dim=(1, 2), keepdim=True) / n

    out = (x - mean) / torch.sqrt(var + eps)

    return out * m


class EnergyAnswerSelector(nn.Module):
    """
    Pairwise (hypothesis x option) energy. Lower energy = better answer.

    Fixed in v45
    ------------
    * **Confidence self-gating removed.** v44 computed
      `energy = energy * (0.5 + confidence)` where `confidence` was a
      head reading the same features as the energy itself. That is a
      multiplicative self-gate: it rescales each (k, n) pair by a
      different learned factor, so energies stop being comparable across
      pairs -- and the marginalization downstream compares them. It has
      no probabilistic interpretation and it is gone. (The earlier form,
      `energy / confidence.clamp(min=0.2)`, was worse: up to 5x
      amplification straight into a +/-6 clamp, which saturated and
      killed gradients -- the documented cause of the reverted warm-start
      experiment.)

    * **The three energies are scale-normalized before fusion.** They
      live on different scales: `learned_energy` is an unbounded MLP
      output, while `hamiltonian_energy` and `cosine_energy` are inner
      products of normalized vectors, bounded in roughly [-1, 1].
      Combining them with per-pair softmax weights meant the learned term
      dominated by magnitude alone, whatever the weights said. Each term
      is now standardized over the (K, N) axes per sample first, so the
      fusion weights actually express a preference.

    * **Alignment and polarity features.** v45 generates a supporting and
      an attacking hypothesis per option. `align[k, n] = 1` when
      hypothesis k is ABOUT option n, and `polarity[k] = +/-1` says which
      direction it argues. Without these the support/attack distinction
      would be a sign bit floating free of the option it refers to, and
      "the objection to option 2" could not be used as evidence against
      option 2 specifically.

    * **Phase head.** Emits theta_kn for models/interference.py, so the
      compatibility amplitude v_kn = t_kn * exp(i theta_kn) is complex
      and the Born rule stops being a reparameterized softmax.

    Kept from v44
    -------------
    Question conditioning (`use_question`) -- the largest measured single
    effect in this project: on cached ARC embeddings a plain MLP scores
    0.3585 from options alone and 0.4849 once the question is available.
    The question text sat in the cache un-encoded for the project's whole
    history.
    """

    def __init__(self, dim, use_question=True):

        super().__init__()

        self.use_question = use_question

        self.hamiltonian = nn.Linear(dim, dim, bias=False)

        # [H, O, H-O, H*O] plus, when available, [Q, Q*O, Q-O];
        # then three scalars: align, polarity, align*polarity.
        n_vec = 7 if use_question else 4

        n_feat = dim * n_vec + 3

        self.energy_net = nn.Sequential(
            nn.Linear(n_feat, dim * 2),
            nn.GELU(),
            nn.LayerNorm(dim * 2),
            nn.Dropout(DROPOUT),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

        self.fusion = nn.Sequential(
            nn.Linear(n_feat, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
        )

        # Deliberately a single linear map rather than an MLP: the phase
        # is one bounded scalar per pair and this keeps the parameter
        # cost of the complex-amplitude extension negligible, so a gain
        # from it cannot be explained as extra capacity.
        self.phase_net = nn.Linear(n_feat, 1)

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, H, O, Q=None, align=None, polarity=None,
                H_mask=None, O_mask=None):
        """
        H        : (B, K, D)
        O        : (B, N, D)
        Q        : (B, D)
        align    : (B, K, N) 1.0 where hypothesis k is about option n
        polarity : (B, K)    +1 support / -1 attack
        """

        B, K, D = H.shape
        _, N, _ = O.shape

        H = F.normalize(H, dim=-1)
        O = F.normalize(O, dim=-1)

        O_proj = self.hamiltonian(O)

        H_exp = H.unsqueeze(2).expand(B, K, N, D)
        O_exp = O_proj.unsqueeze(1).expand(B, K, N, D)

        parts = [H_exp, O_exp, H_exp - O_exp, H_exp * O_exp]

        if self.use_question:

            if Q is None:
                raise ValueError(
                    "EnergyAnswerSelector was built with use_question=True "
                    "but forward() got Q=None. Rebuild the cache so it "
                    "carries question embeddings, or construct the model "
                    "with use_question=False."
                )

            Q_exp = F.normalize(Q, dim=-1).view(B, 1, 1, D).expand(B, K, N, D)

            parts += [Q_exp, Q_exp * O_exp, Q_exp - O_exp]

        if align is None:
            align = torch.zeros(B, K, N, device=H.device, dtype=H.dtype)

        if polarity is None:
            polarity = torch.zeros(B, K, device=H.device, dtype=H.dtype)

        a = align.unsqueeze(-1)
        p = polarity.view(B, K, 1, 1).expand(B, K, N, 1)

        parts += [a, p, a * p]

        features = torch.cat(parts, dim=-1)

        # ------------------------------------------------------------
        # Three energies, standardized onto a common scale
        # ------------------------------------------------------------

        valid = torch.ones(B, K, N, dtype=torch.bool, device=H.device)

        if H_mask is not None:
            valid = valid & H_mask.view(B, K, 1)

        if O_mask is not None:
            valid = valid & O_mask.view(B, 1, N)

        learned_energy = self.energy_net(features).squeeze(-1)

        hamiltonian_energy = -torch.einsum("bkd,bnd->bkn", H, O_proj)

        cosine_energy = -F.cosine_similarity(H_exp, O_exp, dim=-1)

        learned_energy = _masked_standardize(learned_energy, valid)
        hamiltonian_energy = _masked_standardize(hamiltonian_energy, valid)
        cosine_energy = _masked_standardize(cosine_energy, valid)

        fusion_weights = F.softmax(self.fusion(features), dim=-1)

        energy = (
            fusion_weights[..., 0] * learned_energy
            + fusion_weights[..., 1] * hamiltonian_energy
            + fusion_weights[..., 2] * cosine_energy
        )

        temperature = 0.5 + F.softplus(self.temperature)

        energy = energy / temperature

        # Wide safety net against runaway energies. Should not bind in
        # normal training -- if it does, something upstream is diverging.
        energy = torch.clamp(energy, -12, 12)

        # ------------------------------------------------------------
        # Compatibility phase for the coherent collapse
        # ------------------------------------------------------------

        phase = bounded_phase(self.phase_net(features).squeeze(-1))

        return {
            "energy": energy,
            "phase": phase,
            "fusion_weights": fusion_weights,
        }

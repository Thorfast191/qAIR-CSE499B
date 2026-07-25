import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyAnswerSelector(nn.Module):
    """
    Adaptive Hamiltonian Energy Selector.

    Lower Energy = Better Answer.

    Fixed in the audit
    ------------------
    * Question conditioning (`use_question`). This is where the largest
      measured win lives: on cached ARC embeddings, a plain MLP scores
      0.3585 from options alone and 0.4849 once the question is
      available. The question text was in the cache all along but was
      never encoded.
    * `energy = energy / confidence.clamp(min=0.2)` multiplied the energy
      by up to 5x and was then clamped to +/-6. Saturating that clamp
      kills gradients -- the documented cause of the reverted warm-start
      experiment. Confidence now modulates the energy multiplicatively in
      a bounded way (0.5 .. 1.5) and the clamp is a wide safety net
      rather than an active constraint.
    * Dead commented-out normalization block removed.
    """

    def __init__(self, dim, use_question=True):

        super().__init__()

        self.use_question = use_question

        # ------------------------------------------
        # Hamiltonian projection
        # ------------------------------------------

        self.hamiltonian = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        # [H, O, H-O, H*O] plus, when available, [Q, Q*O, Q-O].
        n_feat = 7 if use_question else 4

        # ------------------------------------------
        # Learned compatibility
        # ------------------------------------------

        self.energy_net = nn.Sequential(
            nn.Linear(dim * n_feat, dim * 2),
            nn.GELU(),
            nn.LayerNorm(dim * 2),
            nn.Dropout(0.10),
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

        # ------------------------------------------
        # Adaptive fusion network
        # ------------------------------------------

        self.fusion = nn.Sequential(
            nn.Linear(dim * n_feat, dim),
            nn.GELU(),
            nn.Linear(dim, 3),
        )

        # ------------------------------------------
        # Confidence estimation
        # ------------------------------------------

        self.confidence = nn.Sequential(
            nn.Linear(dim * n_feat, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
            nn.Sigmoid(),
        )

        self.temperature = nn.Parameter(torch.tensor(1.0))

    def forward(self, H, O, Q=None):

        B, K, D = H.shape
        _, N, _ = O.shape

        ##################################################
        # Normalize
        ##################################################

        H = F.normalize(H, dim=-1)
        O = F.normalize(O, dim=-1)

        O_proj = self.hamiltonian(O)

        ##################################################
        # Pairwise tensors
        ##################################################

        H_exp = H.unsqueeze(2).expand(B, K, N, D)
        O_exp = O_proj.unsqueeze(1).expand(B, K, N, D)

        diff = H_exp - O_exp
        prod = H_exp * O_exp

        parts = [H_exp, O_exp, diff, prod]

        if self.use_question:

            if Q is None:
                raise ValueError(
                    "EnergyAnswerSelector was built with use_question=True "
                    "but forward() got Q=None. Rebuild the cache so it "
                    "carries question embeddings (training/dataset.py "
                    "migrates existing caches automatically), or construct "
                    "the model with use_question=False."
                )

            Q_exp = F.normalize(Q, dim=-1).view(B, 1, 1, D).expand(B, K, N, D)

            parts += [Q_exp, Q_exp * O_exp, Q_exp - O_exp]

        features = torch.cat(parts, dim=-1)

        ##################################################
        # Learned Energy
        ##################################################

        learned_energy = self.energy_net(features).squeeze(-1)

        ##################################################
        # Hamiltonian Energy
        ##################################################

        hamiltonian_energy = -torch.einsum(
            "bkd,bnd->bkn",
            H,
            O_proj,
        )

        ##################################################
        # Cosine Energy
        ##################################################

        cosine_energy = -F.cosine_similarity(
            H_exp,
            O_exp,
            dim=-1,
        )

        ##################################################
        # Adaptive Fusion
        ##################################################

        fusion_logits = self.fusion(features)

        fusion_weights = F.softmax(
            fusion_logits,
            dim=-1,
        )

        energy = (

            fusion_weights[..., 0] * learned_energy

            +

            fusion_weights[..., 1] * hamiltonian_energy

            +

            fusion_weights[..., 2] * cosine_energy

        )

        ##################################################
        # Confidence -- bounded modulation in [0.5, 1.5].
        # Sharpens the energy where the model is confident
        # without the 5x blow-up that saturated the clamp.
        ##################################################

        confidence = self.confidence(features).squeeze(-1)

        energy = energy * (0.5 + confidence)

        ##################################################
        # Temperature
        ##################################################

        temperature = 0.5 + F.softplus(
            self.temperature
        )

        energy = energy / temperature

        # Wide safety net against runaway energies. Should not bind in
        # normal training -- if it does, something upstream is diverging.
        energy = torch.clamp(
            energy,
            -12,
            12,
        )

        return {
            "energy": energy,
            "confidence": confidence,
            "fusion_weights": fusion_weights,
        }

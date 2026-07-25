import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PersistentReasoner(nn.Module):
    """
    Persistent Hamiltonian Reasoning.

    Anti-collapse notes (see README "Audit findings")
    ------------------------------------------------
    The pre-audit version of this module contracted to a constant fixed
    point during training: after 5 steps every hypothesis became the same
    vector (measured mean pairwise cos(H_k, H_j) = 1.000000), and that
    vector was independent of the input (||reasoner(H) - reasoner(0)|| =
    1.9e-03). Because every hypothesis was identical, the downstream
    collapse distribution was exactly uniform (1/K) and the model's
    prediction was bit-identically unchanged when H was replaced by
    zeros, by noise, or by another question's hypotheses.

    Three changes address it:

    1. `interaction` is now row-stochastic (softmax over j) instead of
       an unnormalized tanh. The old form computed
       field_k = sum_j a_kj H_j with unnormalized, largely-positive a_kj,
       so field_k ~ (sum_j a_kj) * mean_j(H_j) -- the SAME vector for
       every k. That is textbook oversmoothing, and it was injected with
       a large step size every iteration.

    2. Each step re-anchors to the original hypotheses H0 through a
       gated residual, so the map cannot forget its own input no matter
       how many steps run.

    3. `dt` and `memory_decay` are stored as LOGITS. They used to be
       stored as the post-sigmoid values by mistake: the code wrote
       nn.Parameter(tensor(0.10)) and then applied sigmoid, giving an
       effective step size of sigmoid(0.10) = 0.525 -- 5x the intended
       0.10, which directly accelerated the collapse. Same bug for
       memory_decay: 0.80 became sigmoid(0.80) = 0.690.

    None of this is sufficient on its own -- the collapse is *learned*,
    not present at initialization (measured: pairwise cos 0.51 -> 0.55
    after 5 steps at random init). It is learned because the hypotheses
    carried no usable signal, so discarding them minimized the loss.
    These changes remove the easy path to that solution; fixing the data
    (question conditioning, hypothesis quality) is what removes the
    incentive. Track `pairwise_cos` in training/evaluate.py to confirm.
    """

    def __init__(self, dim, steps=3, keep_trajectory=False):

        super().__init__()

        self.steps = steps

        # trajectory.append(H.detach().cpu()) forces a device->host sync on
        # every step of every forward pass. Only the visualization tools
        # consume it, so it is off during training/eval by default.
        self.keep_trajectory = keep_trajectory

        ####################################################
        # Hamiltonian metric
        ####################################################

        self.metric = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        self.hamiltonian = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        ####################################################
        # Interference network
        ####################################################

        self.interference = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        ####################################################
        # Adaptive gate
        ####################################################

        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        ####################################################
        # Feedforward refinement
        ####################################################

        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(dim * 4, dim),
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        ####################################################
        # Learnable evolution rate -- stored as a LOGIT.
        # logit(0.10) = -2.1972, so sigmoid(dt) starts at 0.10.
        ####################################################

        self.dt = nn.Parameter(torch.tensor(-2.1972))

        ####################################################
        # Validator influence (logit; sigmoid -> 0.30)
        ####################################################

        self.validator_weight = nn.Parameter(torch.tensor(-0.8473))

        ####################################################
        # Persistent memory (logit; sigmoid -> 0.80)
        ####################################################

        self.memory_decay = nn.Parameter(torch.tensor(1.3863))

        ####################################################
        # Anchor back to the input hypotheses (logit; sigmoid -> 0.50)
        ####################################################

        self.anchor_weight = nn.Parameter(torch.tensor(0.0))

    def forward(self, H, potential=None, mask=None):
        """
        H        : (B, K, D) hypothesis embeddings
        potential: (B, K, D) optional validator guidance field
        mask     : (B, K) bool, True for real hypotheses. Padded entries
                   are excluded from the interaction so a zero-padded
                   hypothesis can't drag the field toward the origin.
        """

        H0 = H

        trajectory = []

        interaction = None

        memory = torch.zeros_like(H)

        for _ in range(self.steps):

            ################################################
            # Normalize hypotheses
            ################################################

            H_norm = F.normalize(H, dim=-1)

            ################################################
            # Hamiltonian metric
            ################################################

            H_metric = self.metric(H_norm)

            ################################################
            # Row-stochastic interaction matrix
            ################################################

            logits = torch.einsum(
                "bkd,bjd->bkj",
                H_metric,
                H_norm,
            )

            logits = logits / math.sqrt(H.shape[-1])

            if mask is not None:
                logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))

            interaction = torch.softmax(logits, dim=-1)

            ################################################
            # Hamiltonian field
            ################################################

            field = torch.einsum(
                "bkj,bjd->bkd",
                interaction,
                H,
            )

            field = self.hamiltonian(field)

            ################################################
            # Quantum interference
            ################################################

            field = field + self.interference(field)

            ################################################
            # Persistent memory
            ################################################

            decay = torch.sigmoid(self.memory_decay)

            memory = decay * memory + (1.0 - decay) * field

            field = field + memory

            ################################################
            # Validator guidance
            ################################################

            if potential is not None:

                weight = torch.sigmoid(self.validator_weight)

                field = field + weight * potential

            ################################################
            # Evolution gate
            ################################################

            gate = self.gate(
                torch.cat(
                    [
                        H,
                        field,
                    ],
                    dim=-1,
                )
            )

            ################################################
            # Stable Hamiltonian evolution
            ################################################

            step_size = torch.sigmoid(self.dt)

            H = self.norm1(
                H + gate * step_size * field
            )

            ################################################
            # Refinement
            ################################################

            H = self.norm2(
                H + self.ffn(H)
            )

            ################################################
            # Re-anchor to the original hypotheses so the map
            # cannot converge to an input-independent point.
            ################################################

            H = self.norm3(
                H + torch.sigmoid(self.anchor_weight) * H0
            )

            if self.keep_trajectory:
                trajectory.append(H.detach().cpu())

        return (
            H,
            trajectory,
            interaction,
        )

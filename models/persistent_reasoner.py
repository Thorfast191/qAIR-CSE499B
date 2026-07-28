import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DROPOUT

MIXINGS = ("orthogonal", "softmax")


class PersistentReasoner(nn.Module):
    """
    Persistent multi-hypothesis refinement.

    Naming, honestly
    ----------------
    Earlier versions called this "Persistent Hamiltonian Reasoning". It
    was not Hamiltonian in any sense a physicist would accept:
    `interaction = softmax(metric(H) H^T / sqrt(d))` is not Hermitian (W
    is unconstrained), and `H <- LayerNorm(H + g * dt * field)` is not
    unitary. Unitary evolution U = exp(-i H t) preserves norm and is
    invertible, so it *cannot* map two distinct states onto the same
    one. The implemented map was dissipative and provably contractive:
    it collapsed to a constant fixed point (mean pairwise cosine ->
    1.000000, output independent of input to 1.9e-03).

    What v45 actually fixes
    -----------------------
    The mixing across hypotheses -- the only part that can oversmooth --
    is now an ORTHOGONAL operator, built by the Cayley transform of an
    antisymmetric coupling matrix:

        S = metric(H_norm) H_norm^T / sqrt(d)
        A = (S - S^T) / 2                       (antisymmetric)
        X = dt * A / 2
        U = (I - X)^{-1} (I + X)                (orthogonal, exactly)

    Three consequences, all of them the point:

      * ||U H||_F = ||H||_F and U is invertible, so distinct hypotheses
        stay distinct. The v44 collapse is not merely discouraged, it is
        unreachable through the mixing path.
      * (I - X) is always invertible: the eigenvalues of a real
        antisymmetric matrix are purely imaginary, so 1 - i*lambda is
        never zero. No numerical guard is needed.
      * U's entries are SIGNED. v44's row-softmax made every mixing
        weight non-negative, so field_k was a convex combination of the
        hypotheses and cancellation was arithmetically impossible -- the
        module named "interference" could not interfere. Signed weights
        restore that possibility, and unlike v44's pre-audit unnormalized
        `tanh` (which was signed but unbounded, and is what caused the
        collapse in the first place) an orthogonal operator cannot blow
        up or contract.

    What is NOT claimed
    -------------------
    The per-hypothesis refinement that follows the mixing -- the gated
    residual, the FFN, the LayerNorms -- is a standard residual block and
    is not norm-preserving. Only the hypothesis-mixing operator is
    orthogonal. This block as a whole is not unitary evolution and is not
    described as such anywhere.

    `mixing="softmax"` restores the v44 row-stochastic behaviour as a
    parameter-matched ablation arm.

    Anti-collapse machinery kept from v44
    -------------------------------------
    * Each step re-anchors to the original hypotheses H0 through a gated
      residual, so the map cannot forget its own input.
    * `dt`, `memory_decay`, `validator_weight` and `anchor_weight` are
      stored as LOGITS. They were once stored as post-sigmoid values by
      mistake: `nn.Parameter(tensor(0.10))` then `sigmoid(...)` gives an
      effective step size of 0.525, five times the intended 0.10, which
      directly accelerated the collapse.
    """

    def __init__(self, dim, steps=3, keep_trajectory=False, mixing="orthogonal"):

        super().__init__()

        if mixing not in MIXINGS:
            raise ValueError(f"mixing must be one of {MIXINGS}, got {mixing!r}")

        self.steps = steps

        self.mixing = mixing

        # trajectory.append(H.detach().cpu()) forces a device->host sync on
        # every step of every forward pass. Only the visualization tools
        # consume it, so it is off during training/eval by default.
        self.keep_trajectory = keep_trajectory

        ####################################################
        # Coupling
        ####################################################

        self.metric = nn.Linear(dim, dim, bias=False)

        self.coupling = nn.Linear(dim, dim, bias=False)

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
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(dim * 2, dim),
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

    # ==========================================================
    # MIXING OPERATORS
    # ==========================================================

    def _orthogonal_mix(self, H, H_norm, mask, step_size):
        """
        Returns (mixed, U). U is exactly orthogonal by construction.

        Computed in fp32 regardless of autocast: torch.linalg.solve is
        not implemented for half precision on several backends, and the
        matrices here are tiny (K <= 8), so there is nothing to gain
        from lower precision.
        """

        dtype = H.dtype

        H32 = H.float()

        S = torch.einsum(
            "bkd,bjd->bkj", self.metric(H_norm).float(), H_norm.float()
        ) / math.sqrt(H.shape[-1])

        A = 0.5 * (S - S.transpose(1, 2))

        if mask is not None:
            m = mask.float()
            A = A * m.unsqueeze(1) * m.unsqueeze(2)

        X = 0.5 * step_size.float() * A

        eye = torch.eye(A.shape[1], device=A.device, dtype=A.dtype)
        eye = eye.unsqueeze(0).expand_as(A)

        U = torch.linalg.solve(eye - X, eye + X)

        mixed = torch.bmm(U, H32)

        return mixed.to(dtype), U.to(dtype)

    def _softmax_mix(self, H, H_norm, mask):
        """
        v44 behaviour, kept as an ablation arm. Row-stochastic, hence a
        convex combination: field_k = sum_j a_kj H_j with every a_kj >= 0
        can never cancel, and with an unmasked mean-like row it drives
        every hypothesis toward the same vector.
        """

        logits = torch.einsum(
            "bkd,bjd->bkj", self.metric(H_norm), H_norm
        ) / math.sqrt(H.shape[-1])

        if mask is not None:
            logits = logits.masked_fill(~mask.unsqueeze(1), float("-inf"))

        A = torch.softmax(logits, dim=-1)

        return torch.einsum("bkj,bjd->bkd", A, H), A

    # ==========================================================
    # FORWARD
    # ==========================================================

    def forward(self, H, potential=None, mask=None):
        """
        H        : (B, K, D) hypothesis embeddings
        potential: (B, K, D) optional validator guidance field
        mask     : (B, K) bool, True for real hypotheses. Padded entries
                   are excluded from the mixing so a zero-padded
                   hypothesis can't drag the field toward the origin.
        """

        H0 = H

        trajectory = []

        interaction = None

        memory = torch.zeros_like(H)

        step_size = torch.sigmoid(self.dt)

        for _ in range(self.steps):

            H_norm = F.normalize(H, dim=-1)

            ################################################
            # Mixing across hypotheses
            ################################################

            if self.mixing == "orthogonal":
                field, interaction = self._orthogonal_mix(
                    H, H_norm, mask, step_size
                )
            else:
                field, interaction = self._softmax_mix(H, H_norm, mask)

            field = self.coupling(field)

            ################################################
            # Interference network
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
            # Gated evolution
            ################################################

            gate = self.gate(torch.cat([H, field], dim=-1))

            H = self.norm1(H + gate * step_size * field)

            ################################################
            # Refinement
            ################################################

            H = self.norm2(H + self.ffn(H))

            ################################################
            # Re-anchor to the original hypotheses so the map
            # cannot converge to an input-independent point.
            ################################################

            H = self.norm3(H + torch.sigmoid(self.anchor_weight) * H0)

            if mask is not None:
                H = H * mask.unsqueeze(-1).to(H.dtype)

            if self.keep_trajectory:
                trajectory.append(H.detach().cpu())

        return H, trajectory, interaction

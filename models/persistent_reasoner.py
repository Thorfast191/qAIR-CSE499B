import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PersistentReasoner(nn.Module):

    """
    Persistent Hamiltonian Reasoning

    No attention.
    No softmax.

    Hypotheses evolve through
    Hamiltonian interaction,
    interference,
    and persistent refinement.
    """

    def __init__(self, dim, steps=3):

        super().__init__()

        self.steps = steps

        # Learnable Hamiltonian metric
        self.metric = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        # Hamiltonian evolution
        self.hamiltonian = nn.Linear(
            dim,
            dim,
            bias=False,
        )

        # Learnable interference
        self.interference = nn.Sequential(

            nn.Linear(dim, dim),

            nn.GELU(),

            nn.Linear(dim, dim),
        )

        # Evolution gate
        self.gate = nn.Sequential(

            nn.Linear(dim * 2, dim),

            nn.GELU(),

            nn.Linear(dim, dim),

            nn.Sigmoid(),
        )

        # Refinement network
        self.ffn = nn.Sequential(

            nn.Linear(dim, dim * 4),

            nn.GELU(),

            nn.Dropout(0.10),

            nn.Linear(dim * 4, dim),
        )

        self.norm1 = nn.LayerNorm(dim)

        self.norm2 = nn.LayerNorm(dim)

        self.dt = nn.Parameter(
            torch.tensor(0.10)
        )

    def forward(self, H, potential=None):

        trajectory = []

        interaction = None

        for _ in range(self.steps):

            # -----------------------------------------
            # Normalize hypotheses
            # -----------------------------------------

            H_norm = F.normalize(
                H,
                dim=-1,
            )

            # -----------------------------------------
            # Hamiltonian metric
            # -----------------------------------------

            H_metric = self.metric(
                H_norm
            )

            # -----------------------------------------
            # Hypothesis interaction
            # -----------------------------------------

            interaction = torch.einsum(
                "bkd,bjd->bkj",
                H_metric,
                H_norm,
            )

            interaction = interaction / math.sqrt(
                H.shape[-1]
            )

            # -----------------------------------------
            # Hamiltonian field
            # -----------------------------------------

            field = torch.einsum(
                "bkj,bjd->bkd",
                interaction,
                H,
            )

            field = self.hamiltonian(
                field
            )

            # -----------------------------------------
            # Learnable interference
            # -----------------------------------------

            field = field + self.interference(
                field
            )

            # -----------------------------------------
            # Validator potential
            # -----------------------------------------

            field = field + potential if potential is not None else field

            # -----------------------------------------
            # Adaptive evolution gate
            # -----------------------------------------

            gate = self.gate(

                torch.cat(
                    [
                        H,
                        field,
                    ],
                    dim=-1,
                )

            )

            # -----------------------------------------
            # Persistent evolution
            # -----------------------------------------

            H = self.norm1(

                H

                + gate

                * self.dt

                * field

            )

            # -----------------------------------------
            # Refinement
            # -----------------------------------------

            H = self.norm2(

                H

                + self.ffn(H)

            )

            trajectory.append(
                H.detach().cpu()
            )

        return (
            H,
            trajectory,
            interaction,
        )
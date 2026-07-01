import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PersistentReasoner(nn.Module):
    """
    Persistent Hamiltonian Reasoning

    Improvements
    ------------
    • Stable Hamiltonian interaction
    • Residual reasoning memory
    • Adaptive validator guidance
    • Controlled interference
    """

    def __init__(self, dim, steps=3):

        super().__init__()

        self.steps = steps

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

        ####################################################
        # Learnable evolution rate
        ####################################################

        self.dt = nn.Parameter(torch.tensor(0.10))

        ####################################################
        # Validator influence
        ####################################################

        self.validator_weight = nn.Parameter(torch.tensor(0.30))

        ####################################################
        # Persistent memory
        ####################################################

        self.memory_decay = nn.Parameter(torch.tensor(0.80))

    def forward(self, H, potential=None):

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
            # Stable interaction matrix
            ################################################

            interaction = torch.einsum(
                "bkd,bjd->bkj",
                H_metric,
                H_norm,
            )

            interaction = interaction / math.sqrt(H.shape[-1])

            interaction = torch.tanh(interaction)

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

            trajectory.append(H.detach().cpu())

        return (
            H,
            trajectory,
            interaction,
        )
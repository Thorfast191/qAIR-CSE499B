import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DROPOUT


class ClassicalControlLayer(nn.Module):
    """
    Parameter-matched classical control for QuantumEvolutionLayer.

    Same interface (H -> (q_state, energy)) and the same surrounding
    compress / expand / fusion_gate / correction / energy_head
    architecture -- the only thing that differs is what sits between
    compress and expand. QuantumEvolutionLayer runs a PennyLane circuit
    there; this runs a small classical MLP operating at the same width
    (n_qubits*3, matching the circuit's flat measurement-vector size).

    Used to isolate whether QuantumEvolutionLayer's benefit (if any)
    comes from the quantum circuit's structure (superposition +
    entanglement) specifically, or just from having *some* extra
    nonlinear transform in that position of the pipeline. The quantum
    circuit itself is a tiny fraction of QuantumEvolutionLayer's total
    parameters (54 out of ~1.12M at dim=384, n_qubits=6, n_layers=3),
    so keeping everything else identical and only swapping that one
    small piece keeps total parameter count within a fraction of a
    percent either way -- see the printed counts below.
    """

    def __init__(self, dim, n_qubits=6, n_layers=3, verbose=False):

        super().__init__()

        self.n_qubits = n_qubits

        self.compress = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_qubits),
        )

        bottleneck_dim = n_qubits * 3

        # n_layers of nonlinear transform at the circuit's measurement
        # width: first layer projects n_qubits -> bottleneck_dim, the
        # remaining (n_layers - 1) stay at bottleneck_dim.
        transform_layers = [nn.Linear(n_qubits, bottleneck_dim), nn.GELU()]

        for _ in range(n_layers - 1):
            transform_layers += [
                nn.Linear(bottleneck_dim, bottleneck_dim),
                nn.GELU(),
            ]

        self.classical_transform = nn.Sequential(*transform_layers)

        self.expand = nn.Sequential(
            nn.Linear(bottleneck_dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Dropout(DROPOUT),
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )

        self.correction = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(dim, dim),
        )

        # Matches QuantumEvolutionLayer.phase_gain exactly. The phase
        # itself is read off the classical bottleneck the same way the
        # circuit's is read off its measurements (atan2 of two thirds of
        # the vector), so the two arms differ only in what produced the
        # numbers -- which is the entire point of this control.
        self.phase_gain = nn.Parameter(torch.tensor(1.0))

        self.energy_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

        self.norm = nn.LayerNorm(dim)

        if verbose:
            total_params = sum(p.numel() for p in self.parameters())
            transform_params = sum(
                p.numel() for p in self.classical_transform.parameters()
            )
            print(
                f"[ClassicalControlLayer] total parameters: {total_params:,} "
                f"(classical_transform slot: {transform_params:,})"
            )

    def forward(self, H):

        B, K, D = H.shape

        H_norm = F.normalize(H, dim=-1)

        z = self.compress(H_norm.reshape(B * K, D))

        # Same squashing QuantumEvolutionLayer applies before its circuit
        # (there, to fit PennyLane's rotation-angle range) -- kept
        # identical here so both arms see the exact same input
        # distribution into the part that actually differs.
        z = math.pi * torch.tanh(z)

        measurements = self.classical_transform(z)

        n = self.n_qubits

        x_mean = measurements[:, :n].mean(dim=-1)
        y_mean = measurements[:, n:2 * n].mean(dim=-1)

        phase = self.phase_gain * torch.atan2(y_mean, x_mean)

        q = self.expand(measurements)

        H_flat = H.reshape(B * K, D)

        gate = self.fusion_gate(torch.cat([H_flat, q], dim=-1))

        q = gate * q + (1 - gate) * H_flat

        q = self.norm(q + self.correction(q))

        energy = self.energy_head(q)

        energy = torch.tanh(energy).squeeze(-1)

        q = q.reshape(B, K, D)

        energy = energy.reshape(B, K)

        phase = phase.reshape(B, K)

        return q, energy, phase
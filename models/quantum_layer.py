import math

import pennylane as qml

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumEvolutionLayer(nn.Module):
    """
    Quantum Hamiltonian Evolution Layer

    Improvements
    ------------
    • Stable quantum encoding
    • Residual quantum fusion
    • Learnable quantum correction
    • Bounded quantum energy
    """

    def __init__(self, dim, n_qubits=8, n_layers=3):

        super().__init__()

        self.compress = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_qubits),
        )

        dev = qml.device(
            "default.qubit",
            wires=n_qubits,
        )

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):

            # Superposition
            for i in range(n_qubits):
                qml.Hadamard(wires=i)

            # Angle Encoding
            qml.AngleEmbedding(
                inputs,
                wires=range(n_qubits),
            )

            # Entanglement
            qml.StronglyEntanglingLayers(
                weights,
                wires=range(n_qubits),
            )

            measurements = []

            for i in range(n_qubits):

                measurements.append(qml.expval(qml.PauliX(i)))

                measurements.append(qml.expval(qml.PauliY(i)))

                measurements.append(qml.expval(qml.PauliZ(i)))

            return measurements

        weight_shapes = {
            "weights": (
                n_layers,
                n_qubits,
                3,
            )
        }

        self.quantum = qml.qnn.TorchLayer(
            circuit,
            weight_shapes,
        )

        self.expand = nn.Sequential(
            nn.Linear(
                n_qubits * 3,
                dim,
            ),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

        self.fusion_gate = nn.Sequential(
            nn.Linear(
                dim * 2,
                dim,
            ),
            nn.GELU(),
            nn.Linear(
                dim,
                dim,
            ),
            nn.Sigmoid(),
        )

        self.energy_head = nn.Sequential(
            nn.Linear(
                dim,
                dim // 2,
            ),
            nn.GELU(),
            nn.Linear(
                dim // 2,
                1,
            ),
        )

    def forward(self, H):

        B, K, D = H.shape

        # -----------------------------------------
        # Normalize hypotheses
        # -----------------------------------------

        H_norm = F.normalize(
            H,
            dim=-1,
        )

        # -----------------------------------------
        # Compress
        # -----------------------------------------

        z = self.compress(
            H_norm.reshape(
                B * K,
                D,
            )
        )

        z = torch.tanh(z) * math.pi

        # -----------------------------------------
        # Quantum evolution
        # -----------------------------------------

        q = self.quantum(z)

        q = self.expand(q)

        # -----------------------------------------
        # Residual fusion
        # -----------------------------------------

        H_flat = H.reshape(
            B * K,
            D,
        )

        gate = self.fusion_gate(
            torch.cat(
                [
                    H_flat,
                    q,
                ],
                dim=-1,
            )
        )

        q = gate * q + (1.0 - gate) * H_flat

        # -----------------------------------------
        # Bounded Hamiltonian Energy
        # -----------------------------------------

        energy = torch.tanh(self.energy_head(q)).squeeze(-1)

        # -----------------------------------------

        q = q.reshape(
            B,
            K,
            D,
        )

        energy = energy.reshape(
            B,
            K,
        )

        return q, energy

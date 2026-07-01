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
    • Residual quantum evolution
    • Adaptive fusion
    • Deep Hamiltonian energy
    • Better optimization stability
    """

    def __init__(self, dim, n_qubits=8, n_layers=3):

        super().__init__()

        ####################################################
        # Classical compression
        ####################################################

        self.compress = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_qubits),
        )

        ####################################################
        # Quantum device
        ####################################################

        dev = qml.device(
            "default.qubit",
            wires=n_qubits,
        )

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):

            ################################################
            # Initial Superposition
            ################################################

            for i in range(n_qubits):
                qml.Hadamard(wires=i)

            ################################################
            # Angle Encoding
            ################################################

            qml.AngleEmbedding(
                inputs,
                wires=range(n_qubits),
            )

            ################################################
            # Variational Evolution
            ################################################

            qml.StronglyEntanglingLayers(
                weights,
                wires=range(n_qubits),
            )

            ################################################
            # Measurement
            ################################################

            return [
                qml.expval(qml.PauliX(i))
                for i in range(n_qubits)
            ] + [
                qml.expval(qml.PauliY(i))
                for i in range(n_qubits)
            ] + [
                qml.expval(qml.PauliZ(i))
                for i in range(n_qubits)
            ]

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

        ####################################################
        # Expand back
        ####################################################

        self.expand = nn.Sequential(
            nn.Linear(
                n_qubits * 3,
                dim,
            ),
            nn.GELU(),
            nn.LayerNorm(dim),
        )

        ####################################################
        # Residual gate
        ####################################################

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

        ####################################################
        # Quantum correction
        ####################################################

        self.correction = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        ####################################################
        # Deep Hamiltonian energy
        ####################################################

        self.energy_head = nn.Sequential(

            nn.Linear(dim, dim),

            nn.GELU(),

            nn.LayerNorm(dim),

            nn.Linear(dim, dim//2),

            nn.GELU(),

            nn.Linear(dim//2,1)

        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, H):

        B,K,D = H.shape

        ####################################################
        # Normalize
        ####################################################

        H_norm = F.normalize(
            H,
            dim=-1,
        )

        ####################################################
        # Compress
        ####################################################

        z = self.compress(
            H_norm.reshape(
                B*K,
                D,
            )
        )

        z = math.pi * torch.tanh(z)

        ####################################################
        # Quantum Evolution
        ####################################################

        q = self.quantum(z)

        q = self.expand(q)

        ####################################################
        # Residual fusion
        ####################################################

        H_flat = H.reshape(
            B*K,
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

        q = gate*q + (1-gate)*H_flat

        ####################################################
        # Quantum correction
        ####################################################

        q = self.norm(
            q + self.correction(q)
        )

        ####################################################
        # Hamiltonian energy
        ####################################################

        energy = self.energy_head(q)

        energy = torch.tanh(energy).squeeze(-1)

        ####################################################
        # Reshape
        ####################################################

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
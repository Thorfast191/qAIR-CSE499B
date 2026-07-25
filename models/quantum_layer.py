import math

import pennylane as qml

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantumEvolutionLayer(nn.Module):
    """
    Quantum Hamiltonian Evolution Layer
    """

    def __init__(self, dim, n_qubits=12, n_layers=3):

        super().__init__()

        self.compress = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_qubits),
        )

        # batch_obs=True batches the 3*n_qubits separate expval() observables
        # this circuit returns into fewer statevector passes, instead of
        # re-deriving the state once per observable -- pure execution
        # optimization, no change to what's computed.
        dev = qml.device("lightning.qubit", wires=n_qubits, batch_obs=True)

        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):

            for i in range(n_qubits):
                qml.Hadamard(wires=i)

            qml.AngleEmbedding(inputs, wires=range(n_qubits))

            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

            return (
                [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
                + [qml.expval(qml.PauliY(i)) for i in range(n_qubits)]
                + [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            )

        weight_shapes = {"weights": (n_layers, n_qubits, 3)}

        self.quantum = qml.qnn.TorchLayer(circuit, weight_shapes)

        self.expand = nn.Sequential(
            nn.Linear(n_qubits * 3, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Dropout(0.10),
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
            nn.Dropout(0.10),
            nn.Linear(dim, dim),
        )

        self.energy_head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

        self.norm = nn.LayerNorm(dim)

        # Printed so it can be compared directly against
        # ClassicalControlLayer's printed count at runtime, rather than
        # trusting a hand calculation.
        total_params = sum(p.numel() for p in self.parameters())
        circuit_params = sum(p.numel() for p in self.quantum.parameters())
        print(
            f"[QuantumEvolutionLayer] total parameters: {total_params:,} "
            f"(circuit weights: {circuit_params:,})"
        )

    def forward(self, H):

        B, K, D = H.shape

        H_norm = F.normalize(H, dim=-1)

        z = self.compress(H_norm.reshape(B * K, D))

        z = math.pi * torch.tanh(z)

        q = self.quantum(z)

        q = self.expand(q)

        H_flat = H.reshape(B * K, D)

        gate = self.fusion_gate(torch.cat([H_flat, q], dim=-1))

        q = gate * q + (1 - gate) * H_flat

        q = self.norm(q + self.correction(q))

        energy = self.energy_head(q)

        energy = torch.tanh(energy).squeeze(-1)

        q = q.reshape(B, K, D)

        energy = energy.reshape(B, K)

        return q, energy
import math

import pennylane as qml

import torch
import torch.nn as nn


class QuantumEvolutionLayer(nn.Module):

    def __init__(

        self,

        dim,

        n_qubits=6,

        n_layers=2

    ):

        super().__init__()

        self.compress = nn.Linear(
            dim,
            n_qubits
        )

        dev = qml.device(
            "default.qubit",
            wires=n_qubits
        )

        @qml.qnode(
            dev,
            interface="torch"
        )
        def circuit(inputs, weights):

            for i in range(n_qubits):
                qml.Hadamard(wires=i)

            qml.AngleEmbedding(
                inputs,
                wires=range(n_qubits)
            )

            qml.StronglyEntanglingLayers(
                weights,
                wires=range(n_qubits)
            )

            return [
                qml.expval(
                    qml.PauliZ(i)
                )
                for i in range(n_qubits)
            ]

        weight_shapes = {
            "weights": (
                n_layers,
                n_qubits,
                3
            )
        }

        self.quantum = qml.qnn.TorchLayer(
            circuit,
            weight_shapes
        )

        self.expand = nn.Linear(
            n_qubits,
            dim
        )

    def forward(self, H):

        B, K, D = H.shape

        z = self.compress(

            H.reshape(B * K, D)

        )

        z = torch.tanh(z) * math.pi

        q = self.quantum(z)

        q = self.expand(q)

        q = q.reshape(B, K, D)

        return q
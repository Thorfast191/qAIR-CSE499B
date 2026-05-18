import pennylane as qml
import torch
import torch.nn as nn
import math


class QuantumEvolutionLayer(nn.Module):

    def __init__(self, dim, n_qubits=6, n_layers=2):
        super().__init__()

        self.dim = dim
        self.n_qubits = n_qubits

        self.compress = nn.Linear(dim, n_qubits)

        self.weights = nn.Parameter(
            torch.randn(n_layers, n_qubits, 3) * 0.1
        )

        self.dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="backprop")
        def circuit(x, w):

            for i in range(n_qubits):
                qml.Hadamard(wires=i)

            for i in range(n_qubits):
                qml.RY(x[i], wires=i)
                qml.RZ(x[i], wires=i)

            qml.StronglyEntanglingLayers(w, wires=range(n_qubits))

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.circuit = circuit

        self.expand = nn.Linear(n_qubits, dim)

    def forward(self, H):

        B, K, D = H.shape

        z = self.compress(H.reshape(B * K, D))
        z = torch.tanh(z) * math.pi

        outs = []

        for i in range(z.shape[0]):
            q = self.circuit(z[i], self.weights)
            q = torch.stack(q).float()
            outs.append(q)

        q = torch.stack(outs)

        q = self.expand(q)

        q = q.reshape(B, K, D)

        return q
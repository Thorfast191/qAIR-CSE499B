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

        self.dim = dim

        self.n_qubits = n_qubits

        # ====================================================
        # PROJECTION
        # ====================================================

        self.compress = nn.Linear(

            dim,

            n_qubits

        )

        # ====================================================
        # QUANTUM WEIGHTS
        # ====================================================

        self.weights = nn.Parameter(

            torch.randn(

                n_layers,

                n_qubits,

                3

            ) * 0.1

        )

        # ====================================================
        # DEVICE
        # ====================================================

        self.dev = qml.device(

            "default.qubit",

            wires=n_qubits

        )

        # ====================================================
        # CIRCUIT
        # ====================================================

        @qml.qnode(

            self.dev,

            interface="torch",

            diff_method="backprop"

        )

        def circuit(x, w):

            # --------------------------------------------
            # SUPERPOSITION
            # --------------------------------------------

            for i in range(n_qubits):

                qml.Hadamard(wires=i)

            # --------------------------------------------
            # ENCODING
            # --------------------------------------------

            for i in range(n_qubits):

                qml.RY(x[i], wires=i)

                qml.RZ(x[i], wires=i)

            # --------------------------------------------
            # ENTANGLEMENT
            # --------------------------------------------

            qml.StronglyEntanglingLayers(

                w,

                wires=range(n_qubits)

            )

            return [

                qml.expval(qml.PauliZ(i))

                for i in range(n_qubits)

            ]

        self.circuit = circuit

        # ====================================================
        # EXPAND
        # ====================================================

        self.expand = nn.Linear(

            n_qubits,

            dim

        )

    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, H):

        B, K, D = H.shape

        device = H.device

        # ----------------------------------------------------
        # COMPRESS
        # ----------------------------------------------------

        z = self.compress(

            H.reshape(B * K, D)

        )

        z = torch.tanh(z) * math.pi

        # ----------------------------------------------------
        # QUANTUM
        # ----------------------------------------------------

        outs = []

        for i in range(z.shape[0]):

            # --------------------------------------------
            # MOVE TO CPU
            # --------------------------------------------

            x_cpu = z[i].detach().cpu()

            w_cpu = self.weights.detach().cpu()

            # --------------------------------------------
            # CIRCUIT
            # --------------------------------------------

            q = self.circuit(

                x_cpu,

                w_cpu

            )

            q = torch.tensor(

                q,

                dtype=torch.float32

            )

            outs.append(q)

        q = torch.stack(outs)

        # ----------------------------------------------------
        # MOVE BACK TO ORIGINAL DEVICE
        # ----------------------------------------------------

        q = q.to(device)

        # ----------------------------------------------------
        # EXPAND
        # ----------------------------------------------------

        q = self.expand(q)

        q = q.reshape(B, K, D)

        return q
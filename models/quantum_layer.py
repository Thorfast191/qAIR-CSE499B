import math

import pennylane as qml

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DROPOUT


class QuantumEvolutionLayer(nn.Module):
    """
    Quantum evolution layer: compress -> circuit -> expand, gated back
    into the hypothesis stream.

    v45: the circuit drives a phase
    -------------------------------
    The v44 audit's blunt finding was that this layer was decoration --
    54 circuit parameters out of 10.58M (0.0005%), wrapped in a classical
    MLP that did all the work, and bit-exactly a constant function until
    the AngleEmbedding rotation was fixed. Removing it changed nothing
    measurable, which is not a good place for the component a project is
    named after.

    v45 gives it something only a quantum circuit naturally produces: a
    PHASE. The circuit already measures <X>, <Y> and <Z> per wire, and

        phi = atan2(mean <Y>, mean <X>)

    is the azimuthal angle of the measured Bloch vector -- the physically
    meaningful phase of the state the circuit prepared. That angle is fed
    to models/interference.py as the hypothesis phase, where it decides
    whether two hypotheses reinforce or cancel each other's support for
    an option. The circuit therefore participates in an operation with no
    classical equivalent, rather than contributing a 54-parameter
    perturbation to a residual stream.

    ClassicalControlLayer computes its phase the same way from an equally
    sized classical bottleneck (parameter counts stay matched to the
    scalar), so `A1b_quantum_only` vs `A1c_classical_control_only` still
    isolates the circuit and nothing else.
    """

    def __init__(self, dim, n_qubits=12, n_layers=3, verbose=False):

        super().__init__()

        self.n_qubits = n_qubits

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

            # Hadamard puts every wire in |+>, the +1 eigenstate of Pauli-X.
            # AngleEmbedding's DEFAULT rotation is "X", and RX(t)|+> =
            # exp(-i t/2)|+> -- a global phase, which no expectation value
            # can observe. Written that way (as this layer was until the
            # audit), the circuit's output is bit-exactly independent of
            # `inputs`: measured std across distinct inputs was 0.000e+00,
            # i.e. the whole layer was a learned constant and every quantum
            # ablation run before this fix measured nothing.
            #
            # rotation="Y" keeps the intended "start in uniform
            # superposition" motif while making the encoding observable:
            # RY(t)|+> tilts the Bloch vector off the equator. Measured
            # output std across inputs: 0.123 (vs 0.094 for dropping the
            # Hadamard and keeping RX, 0.089 for RZ).
            for i in range(n_qubits):
                qml.Hadamard(wires=i)

            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")

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

        # Single scalar, matched by ClassicalControlLayer, so the phase
        # channel cannot explain a quantum-vs-classical gap through
        # parameter count.
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

        # Printed so it can be compared directly against
        # ClassicalControlLayer's printed count at runtime, rather than
        # trusting a hand calculation. Opt-in: this used to fire on every
        # model construction, including inside evaluation loops.
        if verbose:
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

        measurements = self.quantum(z)

        # measurements is [<X_0..X_{n-1}>, <Y_0..Y_{n-1}>, <Z_0..Z_{n-1}>].
        # The azimuthal angle of the mean Bloch vector is the phase the
        # circuit actually prepared -- see the class docstring.
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
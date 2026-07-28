import math

import pennylane as qml

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DROPOUT, QUANTUM_DEVICE

DEVICE_CHOICES = ("auto", "lightning.gpu", "lightning.qubit")

# Printed at most once per process, not once per model construction --
# an 8-arm x 5-seed ablation grid builds 40 of these, and 40 identical
# "falling back to CPU" lines would bury the one thing worth knowing.
_reported = set()


def _try_construct(name, n_qubits):

    try:
        # batch_obs batches the 3*n_qubits expval() observables into fewer
        # statevector passes -- pure execution optimization, not a change
        # to what's computed. Retry without it if this particular build
        # doesn't expose the kwarg, rather than treating the absence of a
        # batching optimization as a reason to fail over to a different
        # simulator entirely.
        return qml.device(name, wires=n_qubits, batch_obs=True)
    except TypeError:
        return qml.device(name, wires=n_qubits)


def _build_device(n_qubits, preference, verbose):
    """
    Resolves which PennyLane device actually backs the circuit. See
    config.QUANTUM_DEVICE for what each preference means.

    A fallback here is a device SWAP, not an algorithm change -- lightning
    .gpu and lightning.qubit compute the identical function, exactly. It is
    reported (once) because a training run silently spending 20-30x longer
    than expected because a GPU didn't attach is a footgun worth knowing
    about immediately, not because the choice changes any result.
    """

    if preference not in DEVICE_CHOICES:
        raise ValueError(
            f"QUANTUM_DEVICE must be one of {DEVICE_CHOICES}, got "
            f"{preference!r}"
        )

    order = ["lightning.gpu", "lightning.qubit"] if preference == "auto" else [preference]

    errors = []

    for name in order:

        try:
            dev = _try_construct(name, n_qubits)

        except Exception as e:

            errors.append(f"{name}: {type(e).__name__}: {e}")
            continue

        if name == "lightning.gpu" and "gpu_attached" not in _reported:
            _reported.add("gpu_attached")
            print("[QuantumEvolutionLayer] circuit running on lightning.gpu "
                  "(NVIDIA cuStateVec)")

        elif name == "lightning.qubit" and preference == "auto" and errors:
            key = "cpu_fallback"
            if key not in _reported:
                _reported.add(key)
                print(
                    f"[QuantumEvolutionLayer] lightning.gpu unavailable "
                    f"({errors[-1]}) -- circuit running on lightning.qubit "
                    f"(CPU) instead. Measured 20-30x slower than the GPU "
                    f"device (notebooks/diagnose_quantum_speed.py). If this "
                    f"machine has an NVIDIA GPU, `pip install "
                    f"pennylane-lightning[gpu]`; if not, this is the "
                    f"correct fallback and nothing is wrong."
                )

        if verbose:
            print(f"[QuantumEvolutionLayer] device: {name}")

        return dev, name

    if preference != "auto":
        raise RuntimeError(
            f"QUANTUM_DEVICE={preference!r} was requested explicitly but "
            f"device construction failed: {errors[-1]}\n"
            f"Install it with `pip install pennylane-lightning[gpu]` on a "
            f"CUDA-capable machine, or set config.QUANTUM_DEVICE='auto' to "
            f"fall back to lightning.qubit automatically instead of "
            f"raising."
        )

    raise RuntimeError(
        "No PennyLane device could be constructed, including the CPU "
        "fallback lightning.qubit -- check the pennylane-lightning "
        "install:\n" + "\n".join(errors)
    )


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

    v45 also lets `lightning.gpu` (NVIDIA cuStateVec) back this circuit
    instead of the CPU simulator `lightning.qubit` -- see
    `config.QUANTUM_DEVICE`. Same circuit, same trained function; only the
    hardware executing it changes. Doubling the hypothesis count (support
    + attack) doubled how often this runs per forward call, which is what
    makes the GPU device worth having rather than optional polish.
    """

    def __init__(self, dim, n_qubits=12, n_layers=3, verbose=False,
                 quantum_device=QUANTUM_DEVICE):

        super().__init__()

        self.n_qubits = n_qubits

        self.compress = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, n_qubits),
        )

        dev, self.device_name = _build_device(n_qubits, quantum_device, verbose)

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
                f"(circuit weights: {circuit_params:,}, device: {self.device_name})"
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
"""
One-off diagnostic for the QuantumEvolutionLayer's speed. Run this in the
same environment you train in (same Colab session, after `%cd
/content/qAIR-CSE499B` and `pip install -r requirements.txt`), e.g. as a
notebook cell:

    !python notebooks/diagnose_quantum_speed.py
    !python notebooks/diagnose_quantum_speed.py lightning.qubit   # force CPU
    !python notebooks/diagnose_quantum_speed.py lightning.gpu     # force GPU

With no argument it uses config.QUANTUM_DEVICE ("auto" by default: tries
lightning.gpu, falls back to lightning.qubit). Passing an explicit device
name overrides that for this run only -- useful for printing a direct
CPU-vs-GPU comparison by running the script twice.

It isolates the quantum layer completely from the rest of the model and
dataset -- no cache, no LLM, no ARC data needed -- so it tells you
specifically whether the circuit itself is the bottleneck and whether
batching is actually working, independent of everything else in the
pipeline.
"""

import sys
import time

import torch

from config import N_QUBITS, MODEL_DIM, EMBEDDING_DIM, QUANTUM_DEVICE
from models.quantum_layer import QuantumEvolutionLayer, DEVICE_CHOICES

# v45 runs the model at config.MODEL_DIM, not the raw embedding width --
# the layer is built at whatever width it is given, so time it at the
# width it will actually see.
DIM = MODEL_DIM or EMBEDDING_DIM


def time_forward_backward(layer, batch_rows, iters=5):

    H = torch.randn(batch_rows, 1, DIM, requires_grad=False)

    # warm-up (first call pays circuit compilation/tracing cost -- exclude it)
    # v45: the layer returns (state, energy, phase) -- the phase is the
    # circuit's atan2(<Y>, <X>), which feeds models/interference.py.
    q, e, p = layer(H)
    (q.sum() + e.sum() + p.sum()).backward()

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.time()

    for _ in range(iters):
        layer.zero_grad(set_to_none=True)
        q, e, p = layer(H)
        (q.sum() + e.sum() + p.sum()).backward()

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    elapsed = (time.time() - start) / iters

    return elapsed


if __name__ == "__main__":

    quantum_device_pref = QUANTUM_DEVICE

    if len(sys.argv) > 1:
        quantum_device_pref = sys.argv[1]
        if quantum_device_pref not in DEVICE_CHOICES:
            sys.exit(
                f"unknown device {quantum_device_pref!r}; expected one of "
                f"{DEVICE_CHOICES}"
            )

    # torch_device is where the SURROUNDING classical layers (compress,
    # expand, fusion_gate, ...) run -- separate from quantum_device_pref,
    # which is which PennyLane simulator backs the CIRCUIT ITSELF. A
    # circuit on lightning.gpu still talks to a model living on a CUDA
    # torch_device; they're two different "GPU"s in this sentence, and
    # conflating them is the easiest way to misread this script's output.
    torch_device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"torch_device (classical layers): {torch_device}")
    print(f"n_qubits (from config.py): {N_QUBITS}")

    layer = QuantumEvolutionLayer(
        dim=DIM, n_qubits=N_QUBITS, verbose=True,
        quantum_device=quantum_device_pref,
    ).to(torch_device)

    print(f"circuit device (resolved): {layer.device_name}")

    if layer.device_name == "lightning.qubit" and quantum_device_pref == "auto":
        print(
            "(lightning.gpu did not attach -- see the [QuantumEvolutionLayer] "
            "message above for why. This is the CPU baseline; on a CUDA "
            "machine, `pip install pennylane-lightning[gpu]` and re-run.)"
        )

    # Report what diff_method actually got selected. qnode.diff_method
    # just echoes back the requested value ("best", since the circuit
    # doesn't request anything more specific) -- gradient_fn is what
    # "best" actually resolved to, which is the number that matters.
    qnode = layer.quantum.qnode

    print(f"diff_method requested: {qnode.diff_method}")

    try:
        print(f"gradient_fn resolved to: {qnode.gradient_fn}")
    except AttributeError:
        print("gradient_fn: not available on this PennyLane version "
              "(try qnode.transform_program or qml.about() instead)")

    try:
        print(f"circuit device object: {qnode.device}")
    except AttributeError:
        pass

    print()
    print("batch_rows | avg forward+backward time (s) | time per row (ms)")
    print("-" * 65)

    for batch_rows in [1, 4, 16, 64, 128]:

        t = time_forward_backward(layer, batch_rows)

        per_row_ms = (t / batch_rows) * 1000

        print(f"{batch_rows:>10} | {t:>28.4f} | {per_row_ms:>16.3f}")

    print()
    print("If 'time per row' stays roughly FLAT as batch_rows grows:")
    print("  -> batching/broadcasting is working. Qubit count and gate")
    print("     count (n_layers) are the main levers left.")
    print("If 'time per row' stays roughly CONSTANT total time regardless")
    print("of batch_rows growing (i.e. total time scales ~linearly with")
    print("batch_rows):")
    print("  -> PennyLane is looping per-sample in Python. Batching more")
    print("     data per training step will NOT help -- the fix would be")
    print("     elsewhere (diff_method, PennyLane/lightning version, or")
    print("     restructuring to reduce circuit evaluations per step).")

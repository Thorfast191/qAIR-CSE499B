"""
Shared defaults for the qAIR pipeline.

Single source of truth for values that were previously hardcoded in
multiple places (e.g. embedding dim was 768 in training/dataset.py but
384 in evaluation/sample_inference.py after the encoder model changed).
Import from here instead of re-hardcoding.
"""

import torch


def resolve_device():
    """
    cuda > mps > cpu. Used everywhere instead of hardcoding "cuda" so the
    pipeline runs (falls back gracefully) on CPU-only machines and on
    Apple Silicon Macs instead of crashing outright. Note: the quantum
    circuit (models/quantum_layer.py) always runs on lightning.qubit
    (CPU) regardless of what this resolves to -- PennyLane's TorchLayer
    handles moving data to/from the circuit transparently.
    """

    if torch.cuda.is_available():
        return "cuda"

    # getattr guard: torch.backends.mps does not exist before torch 1.12,
    # and requirements.txt deliberately does not pin a torch version.
    mps = getattr(torch.backends, "mps", None)

    if mps is not None and mps.is_available():
        return "mps"

    return "cpu"

# v41: switched from all-mpnet-base-v2 (768-dim) to all-MiniLM-L6-v2
# (384-dim) and n_qubits 12 -> 6. The quantum circuit's simulation cost
# scales roughly O(n_qubits * 2^n_qubits) -- 12 qubits was the dominant
# per-epoch cost. Changing EMBEDDING_DIM invalidates any cache built
# under the old dim; training/dataset.py refuses to silently load a
# dim-mismatched cache (see its cache-loading dim check).
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

N_QUBITS = 6
PERSISTENT_STEPS = 3

BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 5

# Batch size for offline hypothesis generation during cache building.
# Independent of BATCH_SIZE above -- that one governs the training
# DataLoader, this one governs how many QUESTIONS are queued before a
# generation flush. Since v44 the generator expands each question into
# one prompt per option, so the actual LLM batch is roughly
# GEN_BATCH_SIZE * n_options.
GEN_BATCH_SIZE = 8

# 96 was too small to hold FOUR hypotheses in one completion and silently
# truncated a third of them into template fallbacks -- see the module
# docstring in models/generator.py. Since v44 each completion carries a
# single 15-30 word sentence (~45 tokens at 1.35 tokens/word), so 64 is
# ~50% headroom and parse_one() truncates at 60 words regardless.
#
# This is the main lever on cache-build time. generate() runs until every
# sequence in the batch hits EOS or this cap, so one rambling sample
# drags the whole batch to the ceiling -- lowering the cap bounds that
# directly. Measured ~10.6 s/question on 12 CPU cores at 96 tokens, i.e.
# ~10 h for the 3370-question train split. Build on a GPU if you have
# one; this is 20-30x faster there.
GEN_MAX_NEW_TOKENS = 64

# 0.3 is low for a method whose premise is maintaining DIVERSE
# hypotheses in superposition.
GEN_TEMPERATURE = 0.7

BASE_LR = 5e-4
NEW_LR = 2e-4
WEIGHT_DECAY = 2e-2

SEED = 42

# Seeds for the multi-seed evaluation protocol. Single-run differences
# on ARC's 869-example validation split have SE ~1.6pt, so 1-3 point
# gaps between ablation arms are indistinguishable from seed noise.
SEEDS = (42, 43, 44, 45, 46)

CACHE_DIR = "./cache"
CKPT_DIR = "./ckpt"

# Bumped whenever the cache layout changes in a way that makes older
# files unusable. v33 adds question embeddings ("Q") and per-hypothesis
# is_fallback flags, and switches to per-option generation.
CACHE_VERSION = 33

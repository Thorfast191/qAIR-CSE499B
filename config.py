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


# ==========================================================
# BENCHMARK
# ==========================================================
#
# v45: ARC-Challenge ONLY by default.
#
# Every previous version merged ARC-Challenge with ARC-Easy to triple
# the training pool. That is not defensible: they are distinct published
# benchmarks with different difficulty distributions, every paper reports
# them separately, and merging both makes the numbers incomparable to
# prior work AND confounds every effect with the Easy/Challenge mixture
# ratio. Set to "merged" only if you explicitly want the larger pool for
# training, and say so in any write-up.
#
# Cache files are named after this value (see cache_filename), so a
# Challenge cache and a merged cache cannot silently be mistaken for
# each other -- which is exactly what would have happened with the old
# flat "arc_train.pt" name.
ARC_CONFIG = "ARC-Challenge"  # "ARC-Challenge" | "ARC-Easy" | "merged"

ARC_SPLIT_SIZES = {
    "ARC-Challenge": {"train": 1119, "validation": 299, "test": 1172},
    "ARC-Easy": {"train": 2251, "validation": 570, "test": 2376},
    "merged": {"train": 3370, "validation": 869, "test": 3548},
}


def arc_slug(arc_config=None):

    return (arc_config or ARC_CONFIG).lower().replace("-", "_")


def cache_filename(split, arc_config=None):
    """
    e.g. "arc_challenge_train.pt". The benchmark identity is part of the
    filename on purpose -- see the ARC_CONFIG note above.
    """

    return f"{arc_slug(arc_config)}_{split}.pt"


# ==========================================================
# ENCODER
# ==========================================================

# v41: switched from all-mpnet-base-v2 (768-dim) to all-MiniLM-L6-v2
# (384-dim) and n_qubits 12 -> 6. The quantum circuit's simulation cost
# scales roughly O(n_qubits * 2^n_qubits) -- 12 qubits was the dominant
# per-epoch cost. Changing EMBEDDING_DIM invalidates any cache built
# under the old dim; training/dataset.py refuses to silently load a
# dim-mismatched cache (see its cache-loading dim check).
#
# KNOWN STRUCTURAL LIMIT: everything the generator LLM knows reaches the
# trainable model only through these frozen 6-layer sentence embeddings.
# That bottleneck is very likely a larger constraint on accuracy than
# the entire reasoning architecture. v45 works around it in one specific
# way -- the generator's per-option log-likelihood is cached as an
# explicit scalar feature (see USE_LLM_PRIOR) so at least that part of
# the LLM's judgement is not squeezed through MiniLM.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# ==========================================================
# MODEL WIDTH  (v45: capacity/data ratio)
# ==========================================================
#
# The v44 model was 10,578,677 parameters trained on 800 examples --
# 13,223 parameters per example. Train loss fell 54% over six epochs
# while validation accuracy DECLINED monotonically from its epoch-1 peak:
# it memorized the training set before it could learn any compositional
# use of the hypotheses. That is a hard confound -- at that ratio you
# cannot distinguish "hypotheses don't help" from "the model overfits
# before it learns to use them".
#
# Everything downstream of the input projection now runs at MODEL_DIM
# instead of EMBEDDING_DIM. Cost is quadratic in width across the big
# pairwise MLPs, so 384 -> 128 cuts the model roughly 9x, to ~1.1M.
# Set to None to run at full embedding width (the v44 behaviour).
MODEL_DIM = 128

N_QUBITS = 6
PERSISTENT_STEPS = 3

# Which PennyLane device backs the quantum circuit.
#
#   "auto"            -- try lightning.gpu (NVIDIA cuStateVec) first, fall
#                         back to the CPU simulator lightning.qubit if it
#                         can't be constructed (no GPU, no cuStateVec, or
#                         the plugin just isn't installed). Same circuit,
#                         same function, same trained result either way --
#                         this only changes what executes it. Default.
#   "lightning.gpu"   -- require the GPU device; raises loudly instead of
#                         silently training 20-30x slower on CPU. Use when
#                         a slow run should fail fast rather than limp.
#   "lightning.qubit" -- force the CPU simulator even when a GPU is
#                         present, e.g. for a reproducibility check across
#                         machines without one.
#
# lightning.gpu is NOT in requirements.txt: it needs an NVIDIA GPU, the
# CUDA toolkit, and NVIDIA's cuStateVec library, none of which exist on a
# CPU-only machine (this project's own local dev environment, notably).
# Install it on a CUDA-capable box with:
#     pip install pennylane-lightning[gpu]
# (the v45 Colab notebook does this in its install cell). "auto" is what
# makes the same code correct on both.
QUANTUM_DEVICE = "auto"

DROPOUT = 0.15

# ==========================================================
# HYPOTHESIS EVIDENCE  (v45: the decisive fix)
# ==========================================================
#
# The v44 protocol asked the LLM, for each option, to "state the
# reasoning that would make this proposed answer correct". A competent
# model obliges -- for all four options. It manufactures symmetric
# support, and four equally fluent justifications contain no evidence
# about which option is true. Measured on the clean v44 cache (0%
# fallbacks, alignment verified):
#
#     argmax diag(H . O)               0.2481   (chance = 0.2500)
#     mean pairwise cos(H_k, H_j)      0.6466
#     mean pairwise cos(O_k, O_j)      0.4773   <- hypotheses LESS diverse
#                                                  than the options they
#                                                  were generated from
#
# That is not a bug, it is exactly what the prompt requested, and no
# downstream architecture can recover a signal that was never generated.
#
# v45 generates BOTH directions per option: a supporting hypothesis and
# an attacking one ("state the specific reason this answer is wrong").
# A model that can attack a wrong option more convincingly than a right
# one produces ASYMMETRIC evidence, which is the only thing the
# marginalization has to work with. K = len(GEN_MODES) * n_options.
GEN_MODES = ("support", "attack")

# Batch size for offline hypothesis generation during cache building.
# Independent of BATCH_SIZE below -- that one governs the training
# DataLoader, this one governs how many QUESTIONS are queued before a
# generation flush. Each question expands into len(GEN_MODES) * n_options
# prompts, so the actual LLM batch is ~GEN_BATCH_SIZE * 8.
GEN_BATCH_SIZE = 8

# One 15-30 word sentence is ~45 tokens at 1.35 tokens/word, so 64 is
# ~50% headroom and parse_one() truncates at 60 words regardless.
#
# This is the main lever on cache-build time. generate() runs until every
# sequence in the batch hits EOS or this cap, so one rambling sample
# drags the whole batch to the ceiling -- lowering the cap bounds that
# directly. Build on a GPU if you have one; this is 20-30x faster there.
GEN_MAX_NEW_TOKENS = 64

# 0.3 is low for a method whose premise is maintaining DIVERSE
# hypotheses in superposition.
GEN_TEMPERATURE = 0.7

# ==========================================================
# INTERFERENCE  (v45: making the Born rule non-trivial)
# ==========================================================
#
# With real, non-negative amplitudes, amplitude = exp(-E/2T) followed by
# L2 normalization and squaring is algebraically identical to
# softmax(-E/T) -- verified numerically to fp32 epsilon (5.96e-08). It
# adds no expressive content over a temperature-scaled softmax, and
# |sum_k c_k|^2 = sum_k |c_k|^2 exactly because there are no phases.
# No interference is possible; the module named "interference" could not
# interfere.
#
# v45 gives each hypothesis a learned PHASE. The answer amplitude for
# option n becomes A_n = sum_k c_k v_kn with c_k, v_kn complex, and
# P(n) ∝ |A_n|^2 now carries cross-terms 2*Re(c_i conj(c_j) ...) that
# can be NEGATIVE. Hypotheses can cancel. This is the one place the
# project earns the word "quantum", and it is falsifiable: models/
# interference.py reports `interference_ratio`, and PHASE_MODE="zero"
# / "classical" are parameter-matched controls in the ablation grid.
#
#   "learned"   -- full complex amplitudes, phases from the model
#   "zero"      -- coherent sum with all phases pinned to 0 (constructive
#                  only; isolates "phases" from "coherent summation")
#   "classical" -- diagonal/decohered: P(n) = sum_k |c_k|^2 |v_kn|^2,
#                  i.e. exactly the v44 probability-weighted mixture
PHASE_MODE = "learned"

# v46: `destructive_fraction` is NOT on its own evidence of anything.
#
# With K = 2N matched support/attack pairs it is a step function of one
# scalar, models/full_model.py::polarity_phase. Sweeping that scalar
# through CoherentCollapse with everything else fixed gives 0.000 from 0
# to 1.6 rad, 0.494 at 2.4, 0.994 at 2.8, and 1.000 at 3.0 and above. It
# is initialized at pi -- already past the step -- so it reads 1.0 for an
# untrained model and stays there, and it cannot report the SIGN of the
# effect either: saturating at ~1 is consistent with the coherent ranking
# being far better than the decohered mixture AND with it being far worse.
#
# The test that decides is `coherent_acc` vs `classical_acc` in
# training/evaluate.py -- ranking options by the coherent sum against
# ranking them by the decohered limit of the same amplitudes over the same
# energies. Neither passes through the LLM prior, so the delta is the
# phase channel and nothing else.

# Where the hypothesis phase comes from when PHASE_MODE == "learned":
#   "circuit"  -- atan2(<Y>, <X>) of the quantum layer's own measurements,
#                 so the circuit drives a genuinely quantum operation
#                 instead of being a 54-parameter MLP wrapper
#   "learned"  -- a small classical head on the reasoned hypotheses
# "circuit" silently falls back to "learned" when there is no bottleneck
# layer (backend="none").
PHASE_SOURCE = "circuit"

# Mixing across hypotheses inside PersistentReasoner:
#   "orthogonal" -- Cayley transform of an antisymmetric coupling matrix.
#                   Norm-preserving and invertible, so it provably cannot
#                   contract distinct hypotheses onto one vector, and its
#                   weights are SIGNED so cancellation is possible.
#   "softmax"    -- v44 behaviour: row-stochastic, hence a convex
#                   combination, hence incapable of cancellation. Kept as
#                   an ablation arm, not as a default.
MIXING = "orthogonal"

# Add the generator's cached per-option log-likelihood to the final
# score as an explicit, learned-weight log-prior. This is the audit's
# recommendation #1c and it is the only channel through which the LLM's
# own judgement reaches the answer without passing through the frozen
# 384-d encoder. It MUST be ablated (see training/ablations.py) and
# reported -- a system that wins only because of this feature has not
# demonstrated anything about multi-hypothesis reasoning.
USE_LLM_PRIOR = True

# ==========================================================
# TRAINING
# ==========================================================

BATCH_SIZE = 16
EPOCHS = 30
PATIENCE = 5

BASE_LR = 5e-4
NEW_LR = 2e-4

# Raised from 2e-2 with v45's smaller model: at 800 training examples the
# v44 configuration was still deep in the memorization regime.
WEIGHT_DECAY = 5e-2

LABEL_SMOOTHING = 0.05

SEED = 42

# Seeds for the multi-seed evaluation protocol. Single-run differences on
# ARC-Challenge's 299-example validation split have SE ~2.7pt, so 1-3
# point gaps between ablation arms are indistinguishable from seed noise.
SEEDS = (42, 43, 44, 45, 46)

CACHE_DIR = "./cache"
CKPT_DIR = "./ckpt"

# Bumped whenever the cache layout changes in a way that makes older
# files unusable.
#
#   33 -- added question embeddings ("Q"), per-hypothesis is_fallback
#         flags, per-option generation
#   45 -- support+attack hypotheses (K = 2N), per-hypothesis polarity and
#         option alignment, cached per-option LLM log-likelihood, and the
#         benchmark identity in the filename
#
# A v33 cache CANNOT be migrated to v45: the attacking hypotheses and the
# option log-likelihoods do not exist in it and only the LLM can produce
# them. training/dataset.py refuses to load one rather than silently
# training on half the evidence channels.
CACHE_VERSION = 45

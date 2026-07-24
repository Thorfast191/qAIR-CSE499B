# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

qAIR-vNext: a research prototype that replaces single-path LLM reasoning with persistent, multi-hypothesis, quantum-circuit-inspired reasoning, evaluated on ARC science QA. This is an active research codebase, not production software — no test suite, no linter/build config exist in this repo (nothing to invent here). See `README.md` for the full architecture writeup with diagrams; this file covers what a future agent needs to know before editing code.

## Commands

```bash
pip install -r requirements.txt

python main.py --mode train      # single training run (training/runner.py)
python main.py --mode ablation   # deconfounded ablation grid (training/ablations.py)

# Isolated timing check for the quantum layer only -- no dataset/LLM/cache needed
python notebooks/diagnose_quantum_speed.py
```

The full Colab workflow (Drive mount, clone, install, ablation suite, evaluation, visualization exports) lives in `notebooks/qair_v41_colab.ipynb`.

All modules import with absolute paths from the repo root (`from models.generator import ...`, `from training.dataset import ...`, `from config import ...`) — run scripts from the repo root, not from inside a subpackage.

## Architecture

### Two-stage pipeline: offline data prep vs. online trainable model

1. **Offline, cached to disk, runs once per split** (`training/dataset.py`, `models/generator.py`, `models/encoder.py`): for each ARC question, `HypothesisGenerator` (Qwen2.5-0.5B-Instruct) generates one hypothesis per answer option, then `HypothesisEncoder` (`all-MiniLM-L6-v2`, 384-dim) embeds the hypotheses (`H`) and options (`O`). Result is cached to `cache/arc_{split}.pt`. `QAIRvNext` never sees raw text — only `H`/`O` embedding tensors.
2. **Online, trainable, every forward call** (`models/full_model.py: QAIRvNext.forward(H, O, y)`):
   `H` → `PersistentReasoner` (`persistent_steps` iterations of a Hamiltonian-style field: metric projection, interaction matrix, interference, memory decay, gated evolution) → `QuantumEvolutionLayer` (compress to `n_qubits` rotation angles → PennyLane circuit → expand back, gated-fused into `H`, emits `quantum_energy`) → `HypothesisValidator` (pairwise `H`×`O` features → causal/diversity/specificity/relevance heads → Born-rule probability distribution → `validator_energy`, `potential`) → `EnergyAnswerSelector` (pairwise `H`×`O` energy, three fused methods) → `EnergyFusion` (`models/energy_fusion.py`: learned sigmoid blend of selector/quantum/validator energy into per-hypothesis `collapse_energy`) → `CollapseController` (`models/collapse.py`: second Born-rule application, adaptive per-sample temperature, produces `collapse_probs`) → final answer is a **probability-weighted marginalization** of `answer_energy` over hypotheses (soft, not a hard argmax over hypotheses — superposition survives to the final score).

**Known gap:** `HypothesisValidator`'s `potential` is computed but never fed back into `PersistentReasoner` — `QAIRvNext.forward` calls `self.reasoner(H)` with no `potential` argument, even though `persistent_reasoner.py`'s `forward(self, H, potential=None)` already accepts one. The validator currently only reaches the final answer through `validator_energy` in `EnergyFusion`, not through the reasoning trajectory itself.

### `config.py` is the single source of truth for hyperparameters and device

`EMBEDDING_DIM`/`EMBEDDING_MODEL`, `N_QUBITS`, `PERSISTENT_STEPS`, LRs, `BATCH_SIZE`, `EPOCHS`, `PATIENCE`, `GEN_BATCH_SIZE`, `CACHE_DIR`/`CKPT_DIR` all live here and are imported everywhere (`training/runner.py`, `training/ablations.py`, `models/encoder.py`, `evaluation/sample_inference.py`) rather than hardcoded per entry point. This exists because those values used to drift between files independently (e.g. `sample_inference.py` once hardcoded `dim=384` after the encoder had already moved to 768-dim, causing a `mat1 and mat2 shapes cannot be multiplied` crash at inference time) — **when changing any of these, change `config.py` only.**

`config.py` also has `resolve_device()` (`cuda` > `mps` > `cpu`), used everywhere instead of hardcoding `"cuda"` — `HypothesisGenerator`/`HypothesisEncoder` previously defaulted to `device="cuda"` unconditionally and would crash outright on a CPU-only or Apple Silicon machine.

### Cache correctness guard

`QAIRDataset.__init__` (`training/dataset.py`) checks a loaded cache's embedding dim against `config.EMBEDDING_DIM` before using it and **raises** rather than silently mixing embeddings built with a different encoder/dim. Cache building also batches hypothesis generation + encoding in chunks of `GEN_BATCH_SIZE` (`HypothesisGenerator.generate_batch`) instead of one ARC question at a time — this speeds up the one-time cache build only, not per-epoch training (the quantum layer, not data loading, dominates per-epoch cost — see below).

### Ablation grid is deconfounded by design

`training/ablations.py::ABLATIONS` — each config changes exactly one variable relative to its nearest neighbor (`A1_baseline` → `A1b_quantum_only` → `A2_validator` → `A3_persistent` → `A4_full_hybrid`, toggling `use_quantum`/`use_validator`/`persistent_steps`) so effects can be isolated rather than tangled together. Every config trains from a fresh random init — warm-starting from a "parent" config's converged weights was tried and reverted (it saturated `EnergyAnswerSelector`'s ±6 energy clamp and killed gradient flow through the newly-added component).

### Quantum layer is the dominant cost, and is CPU-bound regardless of GPU

`models/quantum_layer.py` runs on PennyLane's `lightning.qubit` — a CPU statevector simulator (`batch_obs=True`), not GPU-accelerated and not physical hardware (exact, noiseless expectation values, no shot sampling — see the Research Disclaimer in `README.md`). Circuit cost scales roughly `O(n_qubits · 2^n_qubits)` per evaluation and runs once per hypothesis per forward call — this is the dominant per-epoch cost in the model at typical qubit counts, and a discrete/cloud GPU does not meaningfully accelerate it at small qubit counts (6-12). It's forced to run in fp32 outside autocast (`models/full_model.py`) since PennyLane's autograd through `lightning.qubit` isn't mixed-precision safe; the `autocast(device_type=...)`/`GradScaler(...)` calls in `full_model.py`/`training/train.py` are guarded to only ever request `"cuda"` when actually on a CUDA tensor/device, since some PyTorch versions validate `device_type` at construction even when disabled.

### The Born rule is applied twice, for different things

Both `models/validator.py` and `models/collapse.py` implement the same pattern (`amplitude = exp(-E/2T)`, `probability = amplitude²`, temperature-scaled, numerically stabilized by subtracting the max log-amplitude before exponentiating) but over different axes: the validator applies it over hypothesis-vs-option compatibility; the collapse controller applies it over the *fused* per-hypothesis energy to decide how much each hypothesis is trusted in the final answer.

### `visualization/*.py` and `evaluation/sample_inference.py` are manual tools

Not invoked by `main.py` or the training loop — run them from `notebooks/qair_v41_colab.ipynb` or standalone.

## Gotchas

- The local `origin` git remote points at `qAIR-CSE499A`, while this directory, the README, and the notebooks all reference `qAIR-CSE499B` — check `git remote -v` before assuming the clone URL in docs matches `origin`.
- `requirements.txt` does not pin an exact `torch` version — install it appropriately for the target machine's CUDA/CPU/MPS setup before installing the rest.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

qAIR-vNext: a research prototype that replaces single-path LLM reasoning with persistent, multi-hypothesis, quantum-circuit-inspired reasoning, evaluated on ARC science QA. This is an active research codebase, not production software — no test suite, no linter/build config exist in this repo (nothing to invent here). See `README.md` for the full architecture writeup with diagrams; this file covers what a future agent needs to know before editing code.

## Commands

```bash
pip install -r requirements.txt

python main.py --mode train      # single training run (training/runner.py)
python main.py --mode ablation   # deconfounded grid x config.SEEDS (training/ablations.py)

# THE regression test -- exits non-zero if the model ignores its hypotheses.
# Run after every training run; see "The failure this repo is built around".
python -m evaluation.input_ablation --ckpt ckpt/qair_v44_best.pt

python -m evaluation.baselines --mode all          # data ceiling + direct-LLM baseline
python -m evaluation.stats --results ckpt/ablation_results.pt   # CIs + McNemar

# Isolated timing check for the quantum layer only -- no dataset/LLM/cache needed
python notebooks/diagnose_quantum_speed.py
```

## The failure this repo is built around

A v44 audit found that the three mechanisms constituting the research
contribution -- the quantum circuit, multi-hypothesis superposition, and
Born-rule collapse -- were all **provably inert**, while the model
reported a plausible 33.7%. Four independent bugs:

1. **The circuit ignored its input.** `Hadamard` prepares |+>, the +1
   eigenstate of Pauli-X; `AngleEmbedding` defaults to `rotation="X"`;
   RX on an X-eigenstate is a global phase. Output std across distinct
   inputs was **0.000e+00**. Fixed with `rotation="Y"`.
2. **The reasoner collapsed to a constant fixed point** (pairwise
   cos -> 1.000000, input-independent). Unnormalized `tanh` aggregation
   plus a `dt` that was 0.525 rather than the intended 0.10 because it
   was stored post-sigmoid. Fixed with softmax rows, an H0 anchor, and
   logit-space parameters.
3. **The prediction was bit-identical under every corruption of H** --
   zeros, noise, shuffled, another question's hypotheses. Only `O`
   mattered. Now guarded by `evaluation/input_ablation.py`.
4. **33% of cached hypotheses were template fallbacks**, and line-based
   parsing misaligned hypotheses to options. Fixed by generating one
   hypothesis per (question, option) pair in its own completion.

Plus: **the question was never encoded** (+12.6 points once it was), and
`collapse_loss` *rewarded* hypothesis collapse by adding `diversity` and
`spread` with positive coefficients.

**Two rules that follow from this:**

- **Never trust accuracy alone here.** Accuracy could not have revealed
  any of the above. Check `pairwise_cos` in the training log (the trainer
  prints `[COLLAPSE WARNING]` above 0.99) and run `input_ablation.py`.
- **Every pre-v44 quantum/ablation number is void.** Do not compare new
  results against old ones or against numbers in old notebooks.

The current cell-by-cell workflow lives in `notebooks/qAIR_v44.ipynb` (local). `qAIR_v43.ipynb` and the `qair_v4[02]_colab.ipynb` notebooks are **pre-audit and will not run against the current API** — kept as history only.

All modules import with absolute paths from the repo root (`from models.generator import ...`, `from training.dataset import ...`, `from config import ...`) — run scripts from the repo root, not from inside a subpackage.

## Architecture

### Two-stage pipeline: offline data prep vs. online trainable model

1. **Offline, cached to disk, runs once per split** (`training/dataset.py`, `models/generator.py`, `models/encoder.py`): for each ARC question, `HypothesisGenerator` (Qwen2.5-0.5B-Instruct) generates one hypothesis per answer option — **each in its own completion**, so hypothesis↔option alignment is positional and cannot drift — then `HypothesisEncoder` (`all-MiniLM-L6-v2`, 384-dim) embeds the question (`Q`), hypotheses (`H`) and options (`O`). Result is cached to `cache/arc_{split}.pt` with a per-hypothesis `is_fallback` flag. `QAIRvNext` never sees raw text — only `Q`/`H`/`O` embedding tensors. Pre-v33 caches are auto-migrated on load to add `Q` (encoder only, no LLM); migration **cannot** repair hypothesis quality, so `report_quality()` warns if the fallback rate exceeds 10%.
2. **Online, trainable, every forward call** (`models/full_model.py: QAIRvNext.forward(H, O, Q, y, H_mask, O_mask)`):
   `H` → `PersistentReasoner` (`persistent_steps` iterations of a Hamiltonian-style field: metric projection, interaction matrix, interference, memory decay, gated evolution) → `QuantumEvolutionLayer` (compress to `n_qubits` rotation angles → PennyLane circuit → expand back, gated-fused into `H`, emits `quantum_energy`) → `HypothesisValidator` (pairwise `H`×`O` features → causal/diversity/specificity/relevance heads → Born-rule probability distribution → `validator_energy`, `potential`) → `EnergyAnswerSelector` (pairwise `H`×`O` energy, three fused methods) → `EnergyFusion` (`models/energy_fusion.py`: learned sigmoid blend of selector/quantum/validator energy into per-hypothesis `collapse_energy`) → `CollapseController` (`models/collapse.py`: second Born-rule application, adaptive per-sample temperature, produces `collapse_probs`) → final answer is a **probability-weighted marginalization** of `answer_energy` over hypotheses (soft, not a hard argmax over hypotheses — superposition survives to the final score).

**Closed in v44:** `HypothesisValidator`'s `potential` used to be computed and never fed back. `QAIRvNext.forward` now runs a second reasoner pass guided by it (`validator_feedback=True` by default).

### `config.py` is the single source of truth for hyperparameters and device

`EMBEDDING_DIM`/`EMBEDDING_MODEL`, `N_QUBITS`, `PERSISTENT_STEPS`, LRs, `BATCH_SIZE`, `EPOCHS`, `PATIENCE`, `GEN_BATCH_SIZE`, `CACHE_DIR`/`CKPT_DIR` all live here and are imported everywhere (`training/runner.py`, `training/ablations.py`, `models/encoder.py`, `evaluation/sample_inference.py`) rather than hardcoded per entry point. This exists because those values used to drift between files independently (e.g. `sample_inference.py` once hardcoded `dim=384` after the encoder had already moved to 768-dim, causing a `mat1 and mat2 shapes cannot be multiplied` crash at inference time) — **when changing any of these, change `config.py` only.**

`config.py` also has `resolve_device()` (`cuda` > `mps` > `cpu`), used everywhere instead of hardcoding `"cuda"` — `HypothesisGenerator`/`HypothesisEncoder` previously defaulted to `device="cuda"` unconditionally and would crash outright on a CPU-only or Apple Silicon machine.

### Cache correctness guard

`QAIRDataset.__init__` (`training/dataset.py`) checks a loaded cache's embedding dim against `config.EMBEDDING_DIM` before using it and **raises** rather than silently mixing embeddings built with a different encoder/dim. Cache building also batches hypothesis generation + encoding in chunks of `GEN_BATCH_SIZE` (`HypothesisGenerator.generate_batch`) instead of one ARC question at a time — this speeds up the one-time cache build only, not per-epoch training (the quantum layer, not data loading, dominates per-epoch cost — see below).

### Ablation grid is deconfounded by design, and now multi-seed

`training/ablations.py::ABLATIONS` — each config changes exactly one variable relative to its nearest neighbor so effects can be isolated rather than tangled together. Every config trains from a fresh random init — warm-starting from a "parent" config's converged weights was tried and reverted (it saturated `EnergyAnswerSelector`'s energy clamp and killed gradient flow through the newly-added component).

**Single-run ablation numbers are not interpretable here.** ARC's 869-example validation split gives a standard error of ~1.6 points, larger than most gaps this grid produces. The suite sweeps `config.SEEDS` and reports mean ± std; `evaluation/stats.py` adds bootstrap CIs and *paired* McNemar tests (paired, because arms see identical examples — far more powerful than comparing independent proportions).

`A1b_quantum_only` vs `A1c_classical_control_only` is the load-bearing comparison: `ClassicalControlLayer` matches `QuantumEvolutionLayer`'s entire surrounding architecture and total parameter count (the circuit is 54 of ~1.12M), so a gap cannot be attributed to capacity.

### Quantum layer is the dominant cost, and is CPU-bound regardless of GPU

`models/quantum_layer.py` runs on PennyLane's `lightning.qubit` — a CPU statevector simulator (`batch_obs=True`), not GPU-accelerated and not physical hardware (exact, noiseless expectation values, no shot sampling — see the Research Disclaimer in `README.md`). Circuit cost scales roughly `O(n_qubits · 2^n_qubits)` per evaluation and runs once per hypothesis per forward call — this is the dominant per-epoch cost in the model at typical qubit counts, and a discrete/cloud GPU does not meaningfully accelerate it at small qubit counts (6-12). It's forced to run in fp32 outside autocast (`models/full_model.py`) since PennyLane's autograd through `lightning.qubit` isn't mixed-precision safe; the `autocast(device_type=...)`/`GradScaler(...)` calls in `full_model.py`/`training/train.py` are guarded to only ever request `"cuda"` when actually on a CUDA tensor/device, since some PyTorch versions validate `device_type` at construction even when disabled.

### The Born rule is applied twice, for different things — and is currently just softmax

Both `models/validator.py` and `models/collapse.py` implement the same pattern (`amplitude = exp(-E/2T)`, L2-normalize, `probability = amplitude²`, numerically stabilized by subtracting the max log-amplitude before exponentiating) but over different axes: the validator applies it over hypothesis-vs-option compatibility; the collapse controller applies it over the *fused* per-hypothesis energy.

**With real, non-negative amplitudes this is algebraically identical to `softmax(-E/T)`** — verified numerically to fp32 epsilon (5.96e-08). It adds no expressive content over a temperature-scaled softmax. Keep the amplitude form (it is the hook for the complex-amplitude extension, where interference cross-terms would make it genuinely non-classical), but **do not describe it as quantum** in any write-up until that lands. Before v44 the validator L1-normalized instead of L2, which is not the Born rule and silently gave it half the collapse controller's effective temperature.

### Masks are threaded end-to-end — keep them that way

`collate_fn` zero-pads ragged option/hypothesis counts (ARC has some 3- and 5-option questions) and emits `H_mask`/`O_mask`. These were computed and then **never passed to the model**, so padded options participated in every reduction and could be returned as the argmax. Any new module that reduces over the K or N axis must accept and honor the mask.

### `visualization/*.py` and `evaluation/sample_inference.py` are manual tools

Not invoked by `main.py` or the training loop — run them from `notebooks/qAIR_v44.ipynb` or standalone.

## Gotchas

- The local `origin` git remote points at `qAIR-CSE499A`, while this directory, the README, and the notebooks all reference `qAIR-CSE499B` — check `git remote -v` before assuming the clone URL in docs matches `origin`.
- `requirements.txt` does not pin an exact `torch` version — install it appropriately for the target machine's CUDA/CPU/MPS setup before installing the rest.
- `set_seed` (`training/seed.py`) existed for the project's whole history and was **never called from any entry point**, so no run was reproducible. It is now called from `main.py`, `runner.py` and `ablations.py`; keep it that way when adding an entry point, and seed the DataLoader generator too (`seeded_generator`).
- Each run writes `ckpt/{name}_history.pt`. It used to be a single shared `ckpt/history.pt` that all seven ablation configs overwrote in turn.
- `PersistentReasoner.dt`, `memory_decay`, `validator_weight` and `anchor_weight` are stored as **logits** and passed through `sigmoid` at use. Writing `nn.Parameter(torch.tensor(0.10))` here means an effective value of 0.525, which is the bug that accelerated the original collapse.
- The validator-feedback pass runs the reasoner **and the quantum layer twice** per forward. Since the circuit is the dominant CPU cost, `validator_feedback=False` roughly halves per-epoch time when iterating quickly.

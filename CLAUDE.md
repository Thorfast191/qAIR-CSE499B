# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

qAIR-vNext: a research prototype that replaces single-path LLM reasoning with persistent, multi-hypothesis, quantum-circuit-inspired reasoning, evaluated on **ARC-Challenge**. This is an active research codebase, not production software — there is no test suite and no linter/build config, and none should be invented here without being asked. `README.md` has the full architecture writeup with diagrams; this file covers what a future agent needs to know before editing code.

**Current version: v45.** Training happens on **Colab, not locally** — `notebooks/qAIR_v45.ipynb` mounts Drive, clones this repo from GitHub, installs, builds the cache into Drive, and trains there. The local checkout is for editing code.

## Commands

```bash
pip install -r requirements.txt

# ONE-TIME. v45 generates TWO hypotheses per option (support + attack) plus
# an option-scoring pass, so budget ~2x v44. Resumable: autosaves every 50
# samples, saves are atomic, rerun the same command to resume.
python scripts/build_cache.py --splits train validation

#   >>> READ THE [CACHE QUALITY] BLOCK IT PRINTS. <<<
#   If no hypothesis-derived statistic beats chance, stop and change the
#   prompts in models/generator.py. Nothing downstream can recover a signal
#   that was never generated. This is the mistake the project spent a year
#   making.

python main.py --mode train                    # single run (training/runner.py)
python main.py --mode ablation                 # full grid x config.SEEDS
python main.py --mode ablation --configs A3_persistent P0_classical_collapse

# THE regression test -- exits non-zero if the model ignores its hypotheses.
# Run after every training run. Reports PER CHANNEL.
python -m evaluation.input_ablation --ckpt ckpt/qair_v45_best.pt

python -m evaluation.baselines --mode all      # cached direct-LLM + data ceiling
python -m evaluation.stats --results ckpt/ablation_results.pt   # CIs + McNemar

# Isolated timing check for the quantum layer only -- no dataset/LLM/cache
python notebooks/diagnose_quantum_speed.py
```

All modules import with absolute paths from the repo root (`from models.generator import ...`, `from config import ...`) — run scripts from the repo root, not from inside a subpackage.

## The two failures this repo is built around

Read this before trusting any number, including one you produce yourself.

### v44: three mechanisms were provably inert

A full audit found the quantum circuit, multi-hypothesis superposition, and Born-rule collapse were **all provably inert** while the model reported a plausible-looking 33.7%. Four independent causes: the circuit ignored its input entirely (`AngleEmbedding` defaults to `rotation="X"`, and RX on the Hadamard's X-eigenstate is a global phase — output std across distinct inputs was `0.000e+00`, i.e. a learned constant); the reasoner contracted to a constant fixed point (pairwise cos → `1.000000`); the prediction was **bit-identical** under every corruption of `H`; and 33% of cached hypotheses were template fallbacks with broken option alignment. Separately, the question was never encoded (+12.6 points once it was), and `collapse_loss` actively *rewarded* hypothesis collapse. All fixed; all fixes still in the code.

### v45: the generation protocol destroyed the signal by construction

Fixing those four bugs did not make the system work, and a second audit found why. v44 asked the LLM, per option, to "state the reasoning that would make this proposed answer correct". A competent model obliges — for **all four options**. Measured on the clean v44 cache (0% fallbacks, alignment guaranteed by construction, question encoded):

```
argmax diag(H . O)            0.2481     chance = 0.2500
mean pairwise cos(H_k, H_j)   0.6466     vs cos(O_k, O_j) = 0.4773
train loss -54% while val acc declined, epochs 1 -> 6
```

Zero discriminative signal, and hypotheses *less* diverse than the options they were generated from. That single fact explains every downstream observation without needing any of the four v44 bugs: the selector ignored `H` because `H` was uninformative, the collapse was uniform because there was nothing to discriminate, and accuracy tracked option priors because that was the only signal present.

**Three rules follow, and they are the most important content in this file:**

1. **Never trust accuracy alone here.** Accuracy could not have revealed any of the above. Check the cache-quality gate at build time, then `pairwise_cos` / `h_cos` / `phase_effect` / `destructive_fraction` in the training log, then `evaluation/input_ablation.py`.
2. **If the signal gate reports chance, fix `models/generator.py`, not the model.** No architecture recovers information that was never generated. Tuning in response to a dead cache is the specific failure mode this project is a case study in.
3. **Every pre-v45 number is void.** Do not compare new results against old ones or against numbers printed in old notebooks. The cache format is incompatible and is *refused* rather than migrated.

`notebooks/qAIR_v45.ipynb` is the current workflow. `qAIR_v44.ipynb` and earlier **will not run against the v45 API** — the model takes `polarity`/`align`/`llm_logprob`, the quantum layer returns three values, and the cache is rejected. They are kept as history only.

## Architecture

### Two stages: offline data prep vs. online trainable model

**1. Offline, cached to disk, once per split** (`training/dataset.py`, `models/generator.py`, `models/encoder.py`). For each ARC question, `HypothesisGenerator` (Qwen2.5-0.5B-Instruct) generates **two hypotheses per answer option** — a `support` and an `attack` — **each in its own completion**, so hypothesis↔option alignment is positional and cannot drift. It also records the LLM's own length-normalized per-option log-likelihood (`score_options`). `HypothesisEncoder` (`all-MiniLM-L6-v2`, 384-dim) embeds question, hypotheses and options. Everything lands in `cache/arc_challenge_{split}.pt` — **the benchmark name is part of the filename**, so a Challenge cache and a merged cache can never be confused.

Cache schema: `K = 2N` hypotheses stored **mode-major** (all supports in option order, then all attacks), plus `polarity` (+1 support / −1 attack), `hyp_option`, `is_fallback`, `llm_logprob`, `Q`, `H`, `O`, `y`, `source`. `QAIRvNext` never sees raw text.

**2. Online, trainable, every forward call** (`models/full_model.py`):

```
in_proj (384 -> MODEL_DIM)
  -> PersistentReasoner      (orthogonal Cayley mixing)
  -> QuantumEvolutionLayer   (returns state, energy, PHASE)
  -> HypothesisValidator     (-> potential, fed back into the reasoner)
  -> EnergyAnswerSelector    (-> E_kn and a compatibility phase theta_kn)
  -> EnergyFusion            (-> E_k)
  -> CollapseController      (adaptive per-sample temperature)
  -> CoherentCollapse        (P(n) proportional to |sum_k c_k v_kn|^2)
  -> + LLM log-prior (optional, ablatable)
  -> log_softmax
```

### `config.py` is the single source of truth

`ARC_CONFIG`, `EMBEDDING_DIM`/`EMBEDDING_MODEL`, `MODEL_DIM`, `N_QUBITS`, `PERSISTENT_STEPS`, `DROPOUT`, `GEN_MODES`, `PHASE_MODE`, `PHASE_SOURCE`, `MIXING`, `USE_LLM_PRIOR`, LRs, `BATCH_SIZE`, `EPOCHS`, `PATIENCE`, `LABEL_SMOOTHING`, `CACHE_VERSION`, `CACHE_DIR`/`CKPT_DIR`. This file exists because those values used to drift between modules independently — `sample_inference.py` once hardcoded `dim=384` after the encoder had moved to 768-dim, crashing at inference. **When changing any of these, change `config.py` only.**

`resolve_device()` (`cuda` > `mps` > `cpu`) is used everywhere instead of hardcoding `"cuda"`. `cache_filename(split, arc_config)` is the only place cache paths are constructed.

### The v45 mechanisms, and the metric that falsifies each

Everything below is a claim with a number attached. If you change one of these modules, keep its metric working — that pairing *is* the design.

- **`models/interference.py`** — complex amplitudes. `c_k = r_k·e^{iφ_k}`, `v_kn = t_kn·e^{iθ_kn}`, `P(n) ∝ |Σ_k c_k v_kn|²`. Cross-terms can be negative, so hypotheses cancel. `P_n = |v_n⟩⟨v_n|` is Hermitian and positive semi-definite, so `⟨ψ|P_n|ψ⟩` is a legitimate measurement rather than a metaphor.
  - `phase_effect` → 0 ⇒ the phases are inert (coherent sum equals the in-phase sum) and the Born rule is a softmax again.
  - `destructive_fraction` → 0 ⇒ nothing is cancelling. **This is the strict one**: it is identically 0 for any real non-negative amplitude scheme, so a non-zero value is the one measurement here that a classical mixture cannot reproduce.
  - `interference_ratio` is large *even with zero phases* — it only says amplitudes are summed before squaring. Never quote it alone as evidence of anything quantum.
  - `phase_mode ∈ {"learned", "zero", "classical"}` are parameter-matched controls; `"classical"` is exactly v44's mixture.

- **`models/persistent_reasoner.py`** — orthogonal mixing via the Cayley transform of an antisymmetric coupling: `U = (I − X)⁻¹(I + X)`, `X = dt·A/2`, `A = (S − Sᵀ)/2`. Norm-preserving and invertible, so distinct hypotheses provably stay distinct; **signed**, so cancellation is possible (v44's row-softmax was a convex combination and could not cancel). `(I − X)` is always invertible — a real antisymmetric matrix has purely imaginary eigenvalues, so `1 − iλ ≠ 0` — no numerical guard needed. Computed in fp32 (`torch.linalg.solve` has no half-precision kernel on several backends; the matrices are `K×K`, `K ≤ 10`). `mixing="softmax"` restores v44 as an ablation arm.
  - **Only the mixing is orthogonal.** The residual FFN around it is not norm-preserving. Do not describe this block as unitary evolution.

- **`models/quantum_layer.py`** — returns `(state, energy, phase)` where `phase = phase_gain · atan2(mean⟨Y⟩, mean⟨X⟩)`, the azimuthal angle of the measured Bloch vector. This is what gives the 54 circuit parameters a job no classical component can do. `ClassicalControlLayer` reads a phase identically from an equally sized classical bottleneck, so parameter counts stay matched — measured 1,056,382 (quantum) vs 1,057,138 (classical), +0.07%.

- **`models/full_model.py::polarity_phase`** — support and attack start π out of phase, because an objection to option *n* is the negation of the support for option *n*. It is **learnable** (→ 0 recovers in-phase behaviour). Without it the phase channel starts uniform and `destructive_fraction` is exactly 0 at initialization, i.e. the mechanism the v45 claim rests on would begin dead.

- **The LLM log-prior** (`use_llm_prior`) adds the cached per-option log-likelihood to the final score. Legitimate — it is the only channel by which the LLM's judgement bypasses the frozen 384-d encoder — but it can carry the accuracy on its own. **Always** read `input_ablation` alongside accuracy, and report `A3` (prior off) as the reasoning result and `F0` (prior on) as the system's accuracy. Conflating them is how a thin wrapper around an LLM gets published as a reasoning advance.

### Cache guards

`QAIRDataset.__init__` enforces three things and **raises** rather than proceeding:

1. **Dim guard** — cached embedding dim must match `config.EMBEDDING_DIM`.
2. **Version gate** — cache version must be ≥ 45. A v33 cache has no attacking hypotheses and no `llm_logprob`, and neither can be migrated in; only the LLM can produce them.
3. **Build lock** — heartbeat-based, so two concurrent builds can't clobber each other's `raw_index`. Liveness is tracked by timestamp, *not* by `os.kill(pid, 0)`: on Windows that call terminates the process it is asking about.

`report_quality()` is the gate v44 lacked. It prints zero-shot accuracies (support agreement, attack disagreement, support−attack margin, LLM likelihood) against chance, plus hypothesis/option diversity. **A high fallback rate was never the real problem — symmetric evidence was, and it looks fine on every metric v44 had.**

### The ablation grid is deconfounded by design, and multi-seed

`training/ablations.py::ABLATIONS`, three groups, with `DEFAULTS` merged into every arm via `resolve_config`.

- **Group 1 (`A*`)** isolates the reasoning stack with `use_llm_prior=False` throughout — with the prior on, the grid mostly measures how well each arm learns to lean on it.
- **Group 2 (`P0`, `P1`, `PS`, `M0`, `E0`)** each change exactly one v45 mechanism relative to `A3_persistent`.
- **Group 3 (`F0`)** is the deployed system.

Load-bearing contrasts: `A3` vs `P0` (coherent vs decohered), `A3` vs `P1` (phases vs mere coherent summation), `A1b` vs `A1c` (circuit vs parameter-matched MLP), `A3` vs `E0` (does the attack channel earn its keep), `F0` vs `A3` (how much is just the generator).

**Single-run ablation numbers are not interpretable here.** ARC-Challenge's 299-example validation split gives a standard error of ~2.7 points, larger than most gaps this grid produces. The suite sweeps `config.SEEDS` and reports mean ± std; `evaluation/stats.py` adds bootstrap CIs and *paired* McNemar tests (paired, because arms see identical examples in identical order).

### The quantum circuit's device is a config knob, and defaults to auto-detecting a GPU

`models/quantum_layer.py::_build_device` resolves `config.QUANTUM_DEVICE`:

- `"auto"` (default) — try `lightning.gpu` (NVIDIA cuStateVec) first, fall back to the CPU simulator `lightning.qubit` if it can't be constructed (no GPU, `pennylane-lightning[gpu]` not installed, or the plugin fails to attach). Prints once per process on fallback — not once per model, which matters because an 8-arm × 5-seed ablation grid constructs 40 of these.
- `"lightning.gpu"` — require the GPU device explicitly; raises instead of silently training 20-30x slower.
- `"lightning.qubit"` — force CPU even with a GPU present, e.g. for a cross-machine reproducibility check.

Both backends compute the **identical function** — this is a device swap, not an algorithm change, and no result depends on which one ran. `pennylane-lightning[gpu]` is not in `requirements.txt` (it needs CUDA + cuStateVec, absent on a CPU-only machine); the v45 Colab notebook installs it in its §4. `QuantumEvolutionLayer.device_name` holds whichever attached; `ClassicalControlLayer` (the A1c control) has no PennyLane device at all and is unaffected.

Regardless of backend: circuit cost scales roughly `O(n_qubits · 2^n_qubits)` per evaluation and runs once per hypothesis per forward call — and v45 has **twice as many hypotheses** (`K = 2N`), so this got more expensive, not less. It is forced to run in fp32 outside autocast since PennyLane's autograd isn't mixed-precision safe on either lightning.qubit or lightning.gpu (cuStateVec statevectors are fp32/fp64, not fp16/bf16). The `autocast(device_type=...)` / `GradScaler(...)` calls in `full_model.py` are guarded to request `"cuda"` only when actually on a CUDA tensor, because some PyTorch versions validate `device_type` at construction even when disabled — this is unrelated to and doesn't change based on which PennyLane device is attached.

### Masks are threaded end-to-end — keep them that way

`collate_fn` zero-pads ragged option/hypothesis counts (ARC has 3- and 5-option questions) and emits `H_mask`/`O_mask`. These were once computed and then **never passed to the model**, so padded options participated in every reduction and could be returned as the argmax. Any new module that reduces over the K or N axis must accept and honor the mask.

v45 adds `polarity` (B,K), `align` (B,K,N) and `llm_logprob` (B,N) to the batch. `training/evaluate.py::batch_to_model` is the single place that knows the model's input signature — use it rather than unpacking batches by hand, so `evaluate`, `input_ablation` and the notebooks cannot drift.

### `visualization/*.py` and `evaluation/sample_inference.py` are manual tools

Not invoked by `main.py` or the training loop — run them from `notebooks/qAIR_v45.ipynb` or standalone.

## Gotchas

- The repo was renamed **qAIR-CSE499A → qAIR-CSE499B**. GitHub redirects the old URL, so a checkout whose `origin` still says CSE499A pushes fine and only prints a "repository moved" notice. Canonical URL is CSE499B; fix a stale remote with `git remote set-url origin https://github.com/Thorfast191/qAIR-CSE499B.git`. The v45 notebook clones from a `REPO_URL` variable at the top of its §3.
- `requirements.txt` deliberately does not pin `torch` — install it appropriately for the target machine's CUDA/CPU/MPS setup before installing the rest. On Colab, leave the preinstalled torch alone.
- **Do not use `F.cross_entropy(..., label_smoothing=eps)` on `outputs["scores"]`.** Smoothing puts `eps/N` on *every* column including zero-padded ones, which sit at `-inf`, so the loss returns `inf` on any batch containing a 3-option question — and ARC-Challenge has those. `training/losses.py` applies smoothing over `O_mask` only, and `full_model` floors its final `log_softmax` at −30 for the same reason.
- `set_seed` (`training/seed.py`) existed for the project's whole history and was **never called from any entry point**, so no run was reproducible. It is now called from `main.py`, `runner.py` and `ablations.py`; keep it that way when adding an entry point, and seed the DataLoader generator too (`seeded_generator`).
- Each run writes `ckpt/{name}_history.pt`. It used to be one shared `ckpt/history.pt` that every ablation config overwrote in turn, so a resumed run could be handed another config's curves.
- `PersistentReasoner.dt`, `memory_decay`, `validator_weight` and `anchor_weight` are stored as **logits** and passed through `sigmoid` at use. Writing `nn.Parameter(torch.tensor(0.10))` here means an effective value of `sigmoid(0.10) = 0.525` — the bug that accelerated the original collapse.
- `collate_fn` reorders hypotheses when it shuffles options, and that is only correct because the cache is **mode-major** with contiguous blocks of `N`. If you change the cache layout, change that permutation with it — a silent misalignment here reproduces the exact v43 failure.
- The validator-feedback pass runs the reasoner **and the quantum layer twice** per forward. The circuit is the dominant cost regardless of backend, so `validator_feedback=False` roughly halves per-epoch time while iterating.
- `interference_ratio` must **never** become a loss term. Rewarding it would guarantee the number the quantum claim is tested by — the same mistake the old `diversity`/`spread` terms made in reverse.
- `models/quantum_layer.py::_reported` (the "print the GPU fallback message once" set) is module-level and process-global — it resets on every fresh Python process (a new notebook kernel, a new `python main.py` invocation) but stays silent across every model construction *within* one process, including inside an ablation grid. If a fallback message seems to be missing, check whether it already printed earlier in the same session rather than assuming detection silently changed.

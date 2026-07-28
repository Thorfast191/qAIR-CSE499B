import os
import torch

from config import (
    CACHE_DIR,
    CKPT_DIR,
    EPOCHS,
    PATIENCE,
    N_QUBITS,
    WEIGHT_DECAY,
    SEEDS,
    ARC_CONFIG,
    MODEL_DIM,
    EMBEDDING_DIM as DIM,
    resolve_device,
)

from training.train import Trainer
from training.checkpoint import load_or_resume
from training.evaluate import evaluate
from training.runner import build_loaders
from training.seed import set_seed

from models.full_model import QAIRvNext

device = resolve_device()


# Applied to every arm unless the arm overrides it. Note that the LLM
# prior is OFF by default here: it is a strong feature that can carry the
# accuracy on its own, and an architecture grid run with it on measures
# mostly how well each arm learns to lean on it. It gets its own arm
# (F0) instead.
DEFAULTS = {
    "use_quantum": False,
    "use_validator": False,
    "persistent_steps": 3,
    "use_question": True,
    "phase_mode": "learned",
    "phase_source": "circuit",
    "mixing": "orthogonal",
    "use_llm_prior": False,
    "use_attack": True,
}


# Deconfounded grid: each row changes exactly one variable relative to
# its nearest neighbour, so effects can be isolated rather than tangled.
#
# GROUP 1 -- reasoning stack (LLM prior off throughout)
#
#   A0  -> A1  : the question embedding, absent from the pipeline
#                entirely until v44 and worth more than every other
#                variable in this grid combined (0.3585 -> 0.4849 on a
#                plain MLP probe)
#   A1  -> A1b : the bottleneck layer  (validator off, steps fixed)
#   A1b -> A1c : quantum circuit vs. a parameter-matched classical MLP in
#                the same slot. Same total layer parameter count either
#                way -- the circuit is 54 of ~1.1M -- so a gap here
#                cannot be explained as extra capacity. This is the
#                load-bearing comparison for any quantum claim.
#   A1  -> A2  : the validator
#   A2  -> A3  : the bottleneck layer, with the validator on
#   A3  -> A3c : circuit vs. classical control again, full stack
#   A3  -> A4  : persistent_steps 3 -> 5
#
# GROUP 2 -- the v45 mechanisms, each measured against A3
#
#   A3  -> P0  : coherent |sum_k c_k v_kn|^2 vs. the decohered mixture
#                sum_k |c_k|^2 |v_kn|^2. This is THE interference test:
#                P0 is v44's real-amplitude marginalization, parameter-
#                matched (the phase heads still exist, their output is
#                just not used). If A3 does not beat P0, the complex
#                amplitudes are doing nothing.
#   A3  -> P1  : phases pinned to zero -- separates "summing amplitudes
#                before squaring" from "having non-trivial phases",
#                which are two distinct claims.
#   A3  -> PS  : phase from the circuit's own atan2(<Y>, <X>) vs. from a
#                classical head. Tests whether the CIRCUIT specifically
#                contributes to the one genuinely quantum operation.
#   A3  -> M0  : orthogonal (signed, norm-preserving) hypothesis mixing
#                vs. v44's row-softmax, which is a convex combination and
#                therefore incapable of cancellation.
#   A3  -> E0  : support+attack evidence vs. v44's support-only protocol,
#                on the SAME cache (the attack channel is masked out, not
#                regenerated). Measured zero-shot, v44's protocol gave
#                argmax diag(H.O) = 0.2481 against chance 0.2500.
#
# GROUP 3 -- the deployed system
#
#   A3  -> F0  : adds the cached LLM log-prior. Report F0 as the system's
#                accuracy and A3 as the reasoning result; conflating them
#                is how a pipeline that is really a thin wrapper around
#                its own generator gets reported as a reasoning advance.
ABLATIONS = {

    # ---------------- Group 1: reasoning stack ----------------

    "A0_no_question": {"use_question": False},

    "A1_baseline": {},

    "A1b_quantum_only": {"use_quantum": True},

    "A1c_classical_control_only": {"backend": "classical_control"},

    "A2_validator": {"use_validator": True},

    "A3_persistent": {"use_quantum": True, "use_validator": True},

    "A3c_classical_control_persistent": {
        "backend": "classical_control", "use_validator": True,
    },

    "A4_full_hybrid": {
        "use_quantum": True, "use_validator": True, "persistent_steps": 5,
    },

    # ---------------- Group 2: v45 mechanisms ----------------

    "P0_classical_collapse": {
        "use_quantum": True, "use_validator": True, "phase_mode": "classical",
    },

    "P1_phase_zero": {
        "use_quantum": True, "use_validator": True, "phase_mode": "zero",
    },

    "PS_learned_phase_head": {
        "use_quantum": True, "use_validator": True, "phase_source": "learned",
    },

    "M0_softmax_mixing": {
        "use_quantum": True, "use_validator": True, "mixing": "softmax",
    },

    "E0_support_only": {
        "use_quantum": True, "use_validator": True, "use_attack": False,
    },

    # ---------------- Group 3: deployed system ----------------

    "F0_full_with_llm_prior": {
        "use_quantum": True, "use_validator": True, "use_llm_prior": True,
    },
}


def resolve_config(cfg):

    merged = dict(DEFAULTS)
    merged.update(cfg)

    return merged


def build_model(cfg, n_qubits=N_QUBITS):

    cfg = resolve_config(cfg)

    return QAIRvNext(
        dim=DIM,
        model_dim=MODEL_DIM,
        use_quantum=cfg["use_quantum"],
        use_validator=cfg["use_validator"],
        persistent_steps=cfg["persistent_steps"],
        n_qubits=n_qubits,
        backend=cfg.get("backend"),
        use_question=cfg["use_question"],
        phase_mode=cfg["phase_mode"],
        phase_source=cfg["phase_source"],
        mixing=cfg["mixing"],
        use_llm_prior=cfg["use_llm_prior"],
        use_attack=cfg["use_attack"],
    ).to(device)


def run_ablation_suite(
    cache_dir=CACHE_DIR,
    ckpt_dir=CKPT_DIR,
    epochs=EPOCHS,
    patience=PATIENCE,
    n_qubits=N_QUBITS,
    seeds=SEEDS,
    with_test=True,
    arc_config=ARC_CONFIG,
    configs=None,
    train_samples=None,
    val_samples=None,
):
    """
    Runs every config under every seed.

    configs: optional iterable of ABLATIONS keys, for running a subset
    (a smoke run rarely wants all fourteen arms).

    Single-run differences on ARC-Challenge's 299-example validation
    split carry a standard error around 2.7 points, so the 1-3 point gaps
    this grid produces are indistinguishable from seed noise when each
    config is trained once. Results are aggregated as mean +/- std across
    seeds, and per-example test records are kept so evaluation/stats.py
    can run bootstrap CIs and paired McNemar tests between arms.
    """

    names = list(configs) if configs else list(ABLATIONS)

    unknown = [n for n in names if n not in ABLATIONS]

    if unknown:
        raise KeyError(f"unknown ablation config(s): {unknown}")

    results = {}

    for seed in seeds:

        set_seed(seed)

        train_loader, val_loader, test_loader = build_loaders(
            cache_dir, train_samples=train_samples, val_samples=val_samples,
            seed=seed, with_test=with_test, arc_config=arc_config,
        )

        for name in names:

            cfg = ABLATIONS[name]

            run_name = f"{name}_s{seed}"

            print("\n" + "=" * 60)
            print(f"Running {run_name}")
            print(f"Config: {resolve_config(cfg)}")

            set_seed(seed)

            model = build_model(cfg, n_qubits=n_qubits)

            # NOTE: warm-starting from a "parent" ablation's converged
            # weights was tried and reverted -- it saturated the answer
            # selector's energy clamp immediately on the next config,
            # killing gradient flow through the newly-added component.
            # Every config trains from a fresh random init.

            resolved = resolve_config(cfg)

            has_bottleneck = (
                cfg.get("backend") is not None or resolved["use_quantum"]
            )

            wd = (
                WEIGHT_DECAY
                if (has_bottleneck or resolved["use_validator"])
                else WEIGHT_DECAY / 2
            )

            trainer = Trainer(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                ckpt_dir=ckpt_dir,
                name=run_name,
                weight_decay=wd,
            )

            start_epoch, best_acc, resumed_history = load_or_resume(
                trainer, ckpt_dir, run_name, epochs
            )

            if start_epoch >= epochs:
                print(f"[SKIP] {run_name} already finished {epochs} epochs.")
                history = resumed_history or {"best_acc": best_acc, "acc": []}
            else:
                history = trainer.train(
                    epochs=epochs,
                    start_epoch=start_epoch,
                    best_acc=best_acc,
                    patience=patience,
                )

            entry = {
                "config": resolved,
                "seed": seed,
                "best_val_acc": history.get("best_acc", best_acc),
                "final_pairwise_cos": (
                    history["pairwise_cos"][-1] if history.get("pairwise_cos") else None
                ),
                "final_interference": (
                    history["interference_ratio"][-1]
                    if history.get("interference_ratio") else None
                ),
                "history": history,
            }

            if test_loader is not None:

                best_path = os.path.join(ckpt_dir, f"{run_name}_best.pt")

                if os.path.exists(best_path):
                    model.load_state_dict(
                        torch.load(best_path, map_location=device)["model"]
                    )

                test_metrics, records = evaluate(
                    model, test_loader, device, return_records=True
                )

                entry["test_acc"] = test_metrics["acc"]
                entry["test_pairwise_cos"] = test_metrics["pairwise_cos"]
                entry["test_interference"] = test_metrics["interference_ratio"]
                entry["test_phase_effect"] = test_metrics["phase_effect"]
                entry["test_destructive"] = test_metrics["destructive_fraction"]
                entry["test_records"] = records

            results.setdefault(name, []).append(entry)

    summarize(results)

    torch.save(results, os.path.join(ckpt_dir, "ablation_results.pt"))

    return results


def summarize(results):

    print("\n" + "=" * 92)
    print("Ablation Suite Complete -- mean +/- std across seeds")
    print("=" * 92)
    print(f"{'config':<34s} {'val acc':>16s} {'test acc':>16s} "
          f"{'pw cos':>8s} {'phase':>8s} {'destr':>8s}")

    def avg(runs, key):
        vals = [r[key] for r in runs if r.get(key) is not None]
        return f"{sum(vals)/len(vals):.4f}" if vals else "-"

    for name, runs in results.items():

        val = torch.tensor([r["best_val_acc"] for r in runs], dtype=torch.float)

        test = [r["test_acc"] for r in runs if "test_acc" in r]

        val_s = f"{val.mean():.4f}+/-{val.std(unbiased=False):.4f}"

        if test:
            t = torch.tensor(test, dtype=torch.float)
            test_s = f"{t.mean():.4f}+/-{t.std(unbiased=False):.4f}"
        else:
            test_s = "-"

        print(f"{name:<34s} {val_s:>16s} {test_s:>16s} "
              f"{avg(runs, 'test_pairwise_cos'):>8s} "
              f"{avg(runs, 'test_phase_effect'):>8s} "
              f"{avg(runs, 'test_destructive'):>8s}")

    print(
        "\nA pairwise-cos near 1.0 means that arm's hypotheses collapsed and its "
        "accuracy\nreflects option priors only -- treat it as a null result "
        "regardless of the number.\n"
        "phase ~ 0 or destr ~ 0 means that arm's phases are inert: it is running "
        "a classical\nmixture whatever phase_mode says.\n\n"
        "  A3 vs P0   -- does coherent summation beat the decohered mixture?\n"
        "  A3 vs P1   -- do the PHASES contribute, or just the coherent sum?\n"
        "  A1b vs A1c -- does the quantum circuit beat a parameter-matched MLP?\n"
        "  A3 vs E0   -- does the attacking-evidence channel earn its keep?\n"
        "  F0 vs A3   -- how much of the system is just its own generator?"
    )

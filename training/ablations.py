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


# Deconfounded grid: each row changes exactly one variable relative to
# its nearest neighbor, so effects can actually be isolated.
#
#   A1  -> A1b : isolates quantum (steps fixed at 3, validator off)
#   A2  -> A3  : isolates quantum (steps fixed at 3, validator on)
#   A3  -> A4  : isolates persistent_steps (3 -> 5, quantum+validator on)
#   A1b -> A1c : isolates quantum circuit structure vs. a parameter-matched
#                classical MLP in the same slot (see
#                models/classical_control_layer.py) -- same total layer
#                parameter count either way, so a difference here isn't
#                just "more capacity", it's specifically the circuit.
#   A3  -> A3c : same isolation, with validator+persistent also on
#   A0  -> A1  : isolates the question embedding, which was absent from
#                the pipeline entirely until the audit and is worth more
#                than every other variable in this grid combined
#                (0.3585 -> 0.4849 on a plain MLP probe).
#
# IMPORTANT: every quantum result produced before the AngleEmbedding fix
# in models/quantum_layer.py is void. The circuit was bit-exactly
# independent of its input, so "use_quantum=True" added a learned
# CONSTANT, and A1b-vs-A1 measured nothing. Do not compare new numbers
# against old ones.
ABLATIONS = {
    "A0_no_question": {
        "use_quantum": False,
        "use_validator": False,
        "persistent_steps": 3,
        "use_question": False,
    },
    "A1_baseline": {
        "use_quantum": False,
        "use_validator": False,
        "persistent_steps": 3,
    },
    "A1b_quantum_only": {
        "use_quantum": True,
        "use_validator": False,
        "persistent_steps": 3,
    },
    "A1c_classical_control_only": {
        "backend": "classical_control",
        "use_validator": False,
        "persistent_steps": 3,
    },
    "A2_validator": {
        "use_quantum": False,
        "use_validator": True,
        "persistent_steps": 3,
    },
    "A3_persistent": {
        "use_quantum": True,
        "use_validator": True,
        "persistent_steps": 3,
    },
    "A3c_classical_control_persistent": {
        "backend": "classical_control",
        "use_validator": True,
        "persistent_steps": 3,
    },
    "A4_full_hybrid": {
        "use_quantum": True,
        "use_validator": True,
        "persistent_steps": 5,
    },
}


def build_model(cfg, n_qubits=N_QUBITS):

    return QAIRvNext(
        dim=DIM,
        use_quantum=cfg.get("use_quantum", False),
        use_validator=cfg["use_validator"],
        persistent_steps=cfg["persistent_steps"],
        n_qubits=n_qubits,
        backend=cfg.get("backend"),
        use_question=cfg.get("use_question", True),
    ).to(device)


def run_ablation_suite(
    cache_dir=CACHE_DIR,
    ckpt_dir=CKPT_DIR,
    epochs=EPOCHS,
    patience=PATIENCE,
    n_qubits=N_QUBITS,
    seeds=SEEDS,
    with_test=True,
):
    """
    Runs every config under every seed.

    Single-run ablation differences on ARC's 869-example validation split
    have a standard error around 1.6 points, so the 1-3 point gaps this
    grid produces are indistinguishable from seed noise when each config
    is trained once. Results are aggregated as mean +/- std across seeds,
    and per-example test records are kept so evaluation/stats.py can run
    bootstrap CIs and paired McNemar tests between arms.
    """

    results = {}

    for seed in seeds:

        set_seed(seed)

        train_loader, val_loader, test_loader = build_loaders(
            cache_dir, seed=seed, with_test=with_test
        )

        for name, cfg in ABLATIONS.items():

            run_name = f"{name}_s{seed}"

            print("\n" + "=" * 60)
            print(f"Running {run_name}")
            print(f"Config: {cfg}")

            set_seed(seed)

            model = build_model(cfg, n_qubits=n_qubits)

            # NOTE: warm-starting from a "parent" ablation's converged weights
            # was tried and reverted -- it saturated EnergyAnswerSelector's
            # energy clamp immediately on the next config, killing gradient
            # flow through the newly-added component. Every config trains
            # from a fresh random init.

            has_bottleneck = (
                cfg.get("backend") is not None or cfg.get("use_quantum", False)
            )

            wd = WEIGHT_DECAY if (has_bottleneck or cfg["use_validator"]) else 1e-2

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
                "config": cfg,
                "seed": seed,
                "best_val_acc": history.get("best_acc", best_acc),
                "final_pairwise_cos": (
                    history["pairwise_cos"][-1] if history.get("pairwise_cos") else None
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
                entry["test_records"] = records

            results.setdefault(name, []).append(entry)

    summarize(results)

    torch.save(results, os.path.join(ckpt_dir, "ablation_results.pt"))

    return results


def summarize(results):

    print("\n" + "=" * 78)
    print("Ablation Suite Complete -- mean +/- std across seeds")
    print("=" * 78)
    print(f"{'config':<34s} {'val acc':>16s} {'test acc':>16s} {'pw cos':>8s}")

    for name, runs in results.items():

        val = torch.tensor([r["best_val_acc"] for r in runs], dtype=torch.float)

        test = [r["test_acc"] for r in runs if "test_acc" in r]

        cos = [r.get("test_pairwise_cos") for r in runs if r.get("test_pairwise_cos")]

        val_s = f"{val.mean():.4f}+/-{val.std(unbiased=False):.4f}"

        if test:
            t = torch.tensor(test, dtype=torch.float)
            test_s = f"{t.mean():.4f}+/-{t.std(unbiased=False):.4f}"
        else:
            test_s = "-"

        cos_s = f"{sum(cos)/len(cos):.4f}" if cos else "-"

        print(f"{name:<34s} {val_s:>16s} {test_s:>16s} {cos_s:>8s}")

    print(
        "\nA pairwise-cos near 1.0 means that arm's hypotheses collapsed and "
        "its accuracy reflects option priors only -- treat it as a null "
        "result regardless of the number."
    )

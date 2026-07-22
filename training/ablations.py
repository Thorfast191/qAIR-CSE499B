import torch
from functools import partial

from torch.utils.data import DataLoader

from config import (
    CACHE_DIR,
    CKPT_DIR,
    EPOCHS,
    PATIENCE,
    N_QUBITS,
    BATCH_SIZE,
    WEIGHT_DECAY,
)

from training.dataset import (
    QAIRDataset,
    collate_fn,
    DIM,
)

from training.train import Trainer
from training.checkpoint import load_or_resume

from models.full_model import QAIRvNext

device = "cuda" if torch.cuda.is_available() else "cpu"


# Deconfounded grid: each row changes exactly one variable relative to
# its nearest neighbor, so effects can actually be isolated.
#
#   A1  -> A1b : isolates quantum (steps fixed at 3, validator off)
#   A2  -> A3  : isolates quantum (steps fixed at 3, validator on)
#   A3  -> A4  : isolates persistent_steps (3 -> 5, quantum+validator on)
ABLATIONS = {
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
    "A4_full_hybrid": {
        "use_quantum": True,
        "use_validator": True,
        "persistent_steps": 5,
    },
}


def run_ablation_suite(
    cache_dir=CACHE_DIR,
    ckpt_dir=CKPT_DIR,
    epochs=EPOCHS,
    patience=PATIENCE,
    n_qubits=N_QUBITS,
):

    train_ds = QAIRDataset(split="train", max_samples=None, cache_dir=cache_dir)

    val_ds = QAIRDataset(split="validation", max_samples=None, cache_dir=cache_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, shuffle_options=True),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, shuffle_options=False),
    )

    results = {}

    for name, cfg in ABLATIONS.items():

        print("\n" + "=" * 60)
        print(f"Running {name}")
        print(f"Config: {cfg}")

        model = QAIRvNext(
            dim=DIM,
            use_quantum=cfg["use_quantum"],
            use_validator=cfg["use_validator"],
            persistent_steps=cfg["persistent_steps"],
            n_qubits=n_qubits,
        ).to(device)

        # NOTE: warm-starting from a "parent" ablation's converged weights
        # was tried and reverted -- it saturated EnergyAnswerSelector's
        # +/-6 clamp immediately on the next config, killing gradient flow
        # through the newly-added component. Every config trains from a
        # fresh random init.

        wd = WEIGHT_DECAY if (cfg["use_quantum"] or cfg["use_validator"]) else 1e-2

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            ckpt_dir=ckpt_dir,
            name=name,
            weight_decay=wd,
        )

        start_epoch, best_acc, resumed_history = load_or_resume(
            trainer, ckpt_dir, name, epochs
        )

        if start_epoch >= epochs:

            print(f"[SKIP] {name} already finished {epochs} epochs.")

            results[name] = {
                "config": cfg,
                "best_acc": best_acc,
                "history": resumed_history,
            }

            continue

        history = trainer.train(
            epochs=epochs,
            start_epoch=start_epoch,
            best_acc=best_acc,
            patience=patience,
        )

        results[name] = {
            "config": cfg,
            "best_acc": history.get("best_acc", best_acc),
            "final_val_acc": history["acc"][-1] if history["acc"] else None,
            "history": history,
        }

    print("\n" + "=" * 60)
    print("Ablation Suite Complete.")
    print("=" * 60)
    for name, r in results.items():
        print(
            f"{name:20s} best_acc={r['best_acc']:.4f}  "
            f"final_val_acc={r.get('final_val_acc')}"
        )

    return results
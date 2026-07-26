import os
import torch
from functools import partial

from torch.utils.data import DataLoader

from config import (
    CACHE_DIR,
    CKPT_DIR,
    EPOCHS,
    PATIENCE,
    PERSISTENT_STEPS,
    N_QUBITS,
    BATCH_SIZE,
    WEIGHT_DECAY,
    SEED,
    resolve_device,
)

from training.dataset import QAIRDataset, collate_fn
from config import EMBEDDING_DIM as DIM

from training.train import Trainer
from training.checkpoint import load_or_resume
from training.evaluate import evaluate
from training.seed import set_seed, seeded_generator

from models.full_model import QAIRvNext

device = resolve_device()


def build_loaders(cache_dir, train_samples=None, val_samples=None,
                  test_samples=None, seed=SEED, with_test=True):
    """
    ARC ships train / validation / test. Only train and validation were
    ever used, and validation did double duty: it selected the best
    checkpoint AND supplied the reported number, which biases every
    accuracy this project has published. The test split is loaded here so
    the final number can come from data that never touched model
    selection.
    """

    train_ds = QAIRDataset(split="train", max_samples=train_samples, cache_dir=cache_dir)
    val_ds = QAIRDataset(split="validation", max_samples=val_samples, cache_dir=cache_dir)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=seeded_generator(seed),
        collate_fn=partial(collate_fn, shuffle_options=True),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, shuffle_options=False),
    )

    test_loader = None

    if with_test:

        # Do NOT let a routine `--mode train` silently kick off an
        # hours-long LLM generation run for a split the user may not have
        # built yet. ARC's test split is ~3.5k questions; building it from
        # scratch costs as much as train+validation combined. Skip with a
        # loud, actionable message instead.
        test_cache = os.path.join(cache_dir, "arc_test.pt")

        if not os.path.exists(test_cache):
            print(
                f"\n[NO TEST CACHE] {test_cache} does not exist.\n"
                f"  Skipping held-out test evaluation -- validation numbers\n"
                f"  only, which are optimistically biased because validation\n"
                f"  also selects the checkpoint.\n"
                f"  To build it (slow, one-time, needs the LLM):\n"
                f"      python -c \"from training.dataset import QAIRDataset; \"\n"
                f"                 \"QAIRDataset(split='test')\"\n"
            )
        else:
            test_ds = QAIRDataset(
                split="test", max_samples=test_samples, cache_dir=cache_dir
            )
            test_loader = DataLoader(
                test_ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                collate_fn=partial(collate_fn, shuffle_options=False),
            )
            print(f"Test Samples  : {len(test_ds)}")

    print(f"Train Samples : {len(train_ds)}")
    print(f"Val Samples   : {len(val_ds)}")

    return train_loader, val_loader, test_loader


def run_training(
    cache_dir=CACHE_DIR,
    ckpt_dir=CKPT_DIR,
    train_samples=None,
    val_samples=None,
    epochs=EPOCHS,
    persistent_steps=PERSISTENT_STEPS,
    n_qubits=N_QUBITS,
    seed=SEED,
    name="qair_v44",
    with_test=True,
):

    print("=" * 60)
    print("qAIR-v44 TRAINING")
    print("=" * 60)

    set_seed(seed)

    train_loader, val_loader, test_loader = build_loaders(
        cache_dir, train_samples, val_samples, seed=seed, with_test=with_test
    )

    model = QAIRvNext(
        dim=DIM,
        use_quantum=True,
        use_validator=True,
        persistent_steps=persistent_steps,
        n_qubits=n_qubits,
        verbose=True,
    ).to(device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        ckpt_dir=ckpt_dir,
        name=name,
        weight_decay=WEIGHT_DECAY,
    )

    start_epoch, best_acc, _ = load_or_resume(trainer, ckpt_dir, name, epochs)

    print(f"\nTraining from epoch {start_epoch} to {epochs}")

    history = trainer.train(
        epochs=epochs,
        start_epoch=start_epoch,
        best_acc=best_acc,
        patience=PATIENCE,
    )

    # Save the config alongside so inference can't silently load with
    # mismatched persistent_steps / n_qubits ever again.
    config_path = os.path.join(ckpt_dir, f"{name}_config.pt")
    torch.save(
        {
            "use_quantum": True,
            "use_validator": True,
            "persistent_steps": persistent_steps,
            "n_qubits": n_qubits,
            "use_question": True,
            "seed": seed,
        },
        config_path,
    )
    print(f"[CONFIG SAVED] {config_path}")

    # ----------------------------------------------------------
    # HELD-OUT TEST -- reported once, from the best checkpoint,
    # on data that took no part in model selection.
    # ----------------------------------------------------------

    if test_loader is not None:

        best_path = os.path.join(ckpt_dir, f"{name}_best.pt")

        if os.path.exists(best_path):
            model.load_state_dict(torch.load(best_path, map_location=device)["model"])

        test_metrics, records = evaluate(
            model, test_loader, device, return_records=True
        )

        history["test"] = test_metrics
        history["test_records"] = records

        trainer.save_history(history)

        print("\n" + "=" * 60)
        print(f"[{name}] HELD-OUT TEST acc = {test_metrics['acc']:.4f}  "
              f"pairwise_cos = {test_metrics['pairwise_cos']:.4f}")
        print("=" * 60)

    return history

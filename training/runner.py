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


def run_training(
    cache_dir=CACHE_DIR,
    ckpt_dir=CKPT_DIR,
    train_samples=None,
    val_samples=None,
    epochs=EPOCHS,
    persistent_steps=PERSISTENT_STEPS,
    n_qubits=N_QUBITS,
):

    print("=" * 60)
    print("qAIR-V41 TRAINING")
    print("=" * 60)

    train_ds = QAIRDataset(split="train", max_samples=train_samples, cache_dir=cache_dir)

    val_ds = QAIRDataset(split="validation", max_samples=val_samples, cache_dir=cache_dir)

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

    print(f"Train Samples : {len(train_ds)}")
    print(f"Val Samples   : {len(val_ds)}")

    model = QAIRvNext(
        dim=DIM,
        use_quantum=True,
        use_validator=True,
        persistent_steps=persistent_steps,
        n_qubits=n_qubits,
    ).to(device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        ckpt_dir=ckpt_dir,
        name="qair_v41",
        weight_decay=WEIGHT_DECAY,
    )

    start_epoch, best_acc, _ = load_or_resume(trainer, ckpt_dir, "qair_v41", epochs)

    print(f"\nTraining from epoch {start_epoch} to {epochs}")

    history = trainer.train(
        epochs=epochs,
        start_epoch=start_epoch,
        best_acc=best_acc,
        patience=PATIENCE,
    )

    # Save the config alongside so inference can't silently load with
    # mismatched persistent_steps / n_qubits ever again.
    config_path = os.path.join(ckpt_dir, "qair_v41_config.pt")
    torch.save(
        {
            "use_quantum": True,
            "use_validator": True,
            "persistent_steps": persistent_steps,
            "n_qubits": n_qubits,
        },
        config_path,
    )
    print(f"[CONFIG SAVED] {config_path}")

    return history
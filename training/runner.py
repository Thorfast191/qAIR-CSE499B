"""
FIXED RUNNER CONFIGURATION
Changes:
1. Increased batch_size from 8 to 32 for better gradient estimation
2. Larger batch size provides more stable training and better utilization of GPU
3. Larger batches give more accurate statistics for collapse dynamics
"""

import os
import torch
from functools import partial

from torch.utils.data import DataLoader

from training.dataset import (
    QAIRDataset,
    collate_fn,
    DIM,
)

from training.train import Trainer

from models.full_model import QAIRvNextQuantum 

device = "cuda" if torch.cuda.is_available() else "cpu"


def run_training(
    cache_dir="./cache",
    ckpt_dir="./ckpt",
    train_samples=None,
    val_samples=None,
    epochs=20,
):

    print("=" * 60)
    print("qAIR-V39 TRAINING")
    print("=" * 60)

    train_ds = QAIRDataset(
        split="train",
        max_samples=train_samples,
        cache_dir=cache_dir,
    )

    val_ds = QAIRDataset(
        split="validation",
        max_samples=val_samples,
        cache_dir=cache_dir,
    )

    # FIXED: Increased batch_size from 8 to 32
    # Larger batch size provides:
    # - More stable gradient estimates
    # - Better statistics for collapse dynamics
    # - Better GPU utilization
    # - Faster training
    train_loader = DataLoader(
        train_ds,
        batch_size=32,  # INCREASED from 8
        shuffle=True,
        collate_fn=partial(collate_fn, shuffle_options=True),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=32,  # INCREASED from 8
        shuffle=False,
        collate_fn=partial(collate_fn, shuffle_options=False),
    )

    print(f"Train Samples : {len(train_ds)}")
    print(f"Val Samples   : {len(val_ds)}")

    model = QAIRvNextQuantum(
        dim=DIM,
        use_quantum=True,
        use_validator=True,
        persistent_steps=5,
    ).to(device)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        ckpt_dir=ckpt_dir,
        name="qair_v39",
    )

    latest_ckpt = os.path.join(
        ckpt_dir,
        "qair_v39_latest.pt",
    )

    start_epoch = 0
    best_acc = 0.0
    print("\nCheckpoint Search:")
    print(latest_ckpt)
    print("Exists:", os.path.exists(latest_ckpt))

    if os.path.exists(latest_ckpt):

        print("\n[RESUME] Loading checkpoint:")

        ckpt = torch.load(
            latest_ckpt,
            map_location=device,
        )

        model.load_state_dict(ckpt["model"])
        trainer.optim.load_state_dict(ckpt["optimizer"])

        # Restore scheduler
        trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            trainer.optim,
            T_max=epochs,
            eta_min=1e-6,
        )

        if ckpt.get("scheduler") is not None:
            trainer.scheduler.load_state_dict(ckpt["scheduler"])

        # Restore GradScaler
        if ckpt.get("scaler") is not None:
            trainer.scaler.load_state_dict(ckpt["scaler"])

        # Restore best accuracy
        best_acc = ckpt.get("best_acc", 0.0)

        start_epoch = ckpt["epoch"] + 1

        print(f"Resuming from epoch {start_epoch}")

    print(f"\nTraining from epoch " f"{start_epoch} to {epochs}")
    history = trainer.train(
        epochs=epochs,
        start_epoch=start_epoch,
        best_acc=best_acc,
        patience=5,
    )
    return history

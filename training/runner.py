import torch

from torch.utils.data import DataLoader

from training.dataset import (
    QAIRDataset,
    collate_fn,
    DIM
)

from training.train import Trainer

from models.full_model import QAIRvNext

# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# TRAINING
# ============================================================

def run_training(

    benchmark="arc",

    epochs=5,

    max_samples=500

):

    print("=" * 60)
    print("TRAINING PIPELINE")
    print("=" * 60)

    # ========================================================
    # DATASET
    # ========================================================

    train_ds = QAIRDataset(

        benchmark=benchmark,

        split="train",

        max_samples=max_samples

    )

    val_ds = QAIRDataset(

        benchmark=benchmark,

        split="validation",

        max_samples=max_samples // 4

    )

    # ========================================================
    # LOADERS
    # ========================================================

    train_loader = DataLoader(

        train_ds,

        batch_size=8,

        shuffle=True,

        collate_fn=collate_fn

    )

    val_loader = DataLoader(

        val_ds,

        batch_size=8,

        shuffle=False,

        collate_fn=collate_fn

    )

    print(f"Train Samples : {len(train_ds)}")
    print(f"Val Samples   : {len(val_ds)}")

    # ========================================================
    # MODEL
    # ========================================================

    model = QAIRvNext(

        dim=DIM,

        use_quantum=True,

        use_validator=True,

        persistent_steps=3

    ).to(device)

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = Trainer(

        model=model,

        train_loader=train_loader,

        val_loader=val_loader,

        device=device,

        ckpt_dir="./ckpt",

        meta_dir="./meta",

        name=f"qair_{benchmark}"

    )

    # ========================================================
    # TRAIN
    # ========================================================

    trainer.train(epochs=epochs)
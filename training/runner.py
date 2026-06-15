import torch

from torch.utils.data import DataLoader

from training.dataset import QAIRDataset, collate_fn, DIM

from training.train import Trainer

from models.full_model import QAIRvNext

device = "cuda" if torch.cuda.is_available() else "cpu"


def run_training(
    cache_dir="./cache", ckpt_dir="./ckpt", train_samples=500, val_samples=100, epochs=5
):

    print("=" * 60)
    print("qAIR-V28 TRAINING")
    print("=" * 60)

    # ========================================================
    # DATASETS
    # ========================================================

    train_ds = QAIRDataset(
        split="train", max_samples=train_samples, cache_dir=cache_dir
    )

    val_ds = QAIRDataset(
        split="validation", max_samples=val_samples, cache_dir=cache_dir
    )

    # ========================================================
    # LOADERS
    # ========================================================

    train_loader = DataLoader(
        train_ds, batch_size=8, shuffle=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate_fn)

    print(f"Train Samples : {len(train_ds)}")

    print(f"Val Samples   : {len(val_ds)}")

    # ========================================================
    # MODEL
    # ========================================================

    model = QAIRvNext(
        dim=DIM, use_quantum=True, use_validator=True, persistent_steps=3
    ).to(device)

    # ========================================================
    # TRAINER
    # ========================================================

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        ckpt_dir=ckpt_dir,
        name="qair_v28",
    )

    # ========================================================
    # TRAIN
    # ========================================================

    trainer.train(epochs=epochs)

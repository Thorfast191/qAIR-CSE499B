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

from models.full_model import QAIRvNext

device = "cuda" if torch.cuda.is_available() else "cpu"


ABLATIONS = {
    "A1_baseline": {
        "use_quantum": False,
        "use_validator": False,
        "persistent_steps": 1,
    },
    "A8_full_hybrid": {
        "use_quantum": True,
        "use_validator": True,
        "persistent_steps": 3,
    },
    "A9_validator": {
        "use_quantum": False,
        "use_validator": True,
        "persistent_steps": 3,
    },
    "A10_persistent": {
        "use_quantum": True,
        "use_validator": True,
        "persistent_steps": 5,
    },
}


def run_ablation_suite(
    cache_dir,
    ckpt_dir,
    epochs=20,
):

    train_ds = QAIRDataset(
        split="train",
        max_samples=8000,
        cache_dir=cache_dir,
    )

    val_ds = QAIRDataset(
        split="validation",
        max_samples=2000,
        cache_dir=cache_dir,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        collate_fn=partial(collate_fn, shuffle_options=True),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=partial(collate_fn, shuffle_options=False),
    )

    results = {}

    for name, cfg in ABLATIONS.items():

        print("\n" + "=" * 60)
        print(f"Running {name}")

        model = QAIRvNext(
            dim=DIM,
            use_quantum=cfg["use_quantum"],
            use_validator=cfg["use_validator"],
            persistent_steps=cfg["persistent_steps"],
        ).to(device)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            ckpt_dir=ckpt_dir,
            name=name,
        )

        latest_ckpt = os.path.join(
            ckpt_dir,
            f"{name}_latest.pt",
        )

        start_epoch = 0

        print("\nCheckpoint Search:")
        print(latest_ckpt)
        print("Exists:", os.path.exists(latest_ckpt))

        if os.path.exists(latest_ckpt):

            print(f"\n[RESUME {name}]")

            ckpt = torch.load(
                latest_ckpt,
                map_location=device,
            )

            model.load_state_dict(ckpt["model"])

            trainer.optim.load_state_dict(ckpt["optimizer"])

            # -------------------------

            # Restore scheduler

            # -------------------------

            trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                trainer.optim,
                T_max=epochs,
                eta_min=1e-6,
            )

            if ckpt.get("scheduler") is not None:

                trainer.scheduler.load_state_dict(ckpt["scheduler"])

            # -------------------------

            # Restore AMP GradScaler

            # -------------------------

            if ckpt.get("scaler") is not None:

                trainer.scaler.load_state_dict(ckpt["scaler"])

            start_epoch = ckpt["epoch"] + 1

            print(f"Resuming from epoch " f"{start_epoch}")

        if start_epoch >= epochs:

            print(f"[SKIP] {name} already " f"finished {epochs} epochs.")

            results[name] = cfg

            continue

        trainer.train(
            epochs=epochs,
            start_epoch=start_epoch,
        )

        results[name] = cfg

    print("\nAblation Suite Complete.")

    return results
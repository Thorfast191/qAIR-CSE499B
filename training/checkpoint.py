import os
import torch


def load_or_resume(trainer, ckpt_dir, name, epochs):
    """
    Loads f"{name}_latest.pt" if present and restores trainer state in
    place (model, optimizer, scheduler, scaler). Shared by runner.py and
    ablations.py so the two entry points can't drift on resume behavior.

    Returns (start_epoch, best_acc, resumed_history). resumed_history is
    the contents of history.pt if present, else None.
    """

    latest_ckpt = os.path.join(ckpt_dir, f"{name}_latest.pt")

    start_epoch = 0
    best_acc = 0.0
    resumed_history = None

    print("\nCheckpoint Search:")
    print(latest_ckpt)
    print("Exists:", os.path.exists(latest_ckpt))

    if not os.path.exists(latest_ckpt):
        return start_epoch, best_acc, resumed_history

    print(f"\n[RESUME {name}]")

    ckpt = torch.load(latest_ckpt, map_location=trainer.device)

    trainer.model.load_state_dict(ckpt["model"])
    trainer.optim.load_state_dict(ckpt["optimizer"])

    trainer.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        trainer.optim, T_max=epochs, eta_min=1e-6,
    )

    if ckpt.get("scheduler") is not None:
        trainer.scheduler.load_state_dict(ckpt["scheduler"])

    if ckpt.get("scaler") is not None:
        trainer.scaler.load_state_dict(ckpt["scaler"])

    best_acc = ckpt.get("best_acc", 0.0)
    start_epoch = ckpt["epoch"] + 1

    print(f"Resuming from epoch {start_epoch}")

    history_path = os.path.join(ckpt_dir, "history.pt")
    if os.path.exists(history_path):
        try:
            resumed_history = torch.load(history_path, map_location="cpu")
        except Exception as e:
            print(f"[WARN] Could not load history.pt: {e}")

    return start_epoch, best_acc, resumed_history

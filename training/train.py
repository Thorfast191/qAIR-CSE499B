import os
import torch

from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler

from training.losses import compute_loss
from training.evaluate import evaluate


class Trainer:

    def __init__(self, model, train_loader, val_loader, device, ckpt_dir, name):

        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device

        self.ckpt_dir = ckpt_dir
        self.name = name

        os.makedirs(ckpt_dir, exist_ok=True)

        self.optim = torch.optim.AdamW(
            model.parameters(), lr=1e-4, weight_decay=1e-2, betas=(0.9, 0.98)
        )

        self.scaler = GradScaler("cuda", enabled=(device == "cuda"))

        self.scheduler = None

    # =====================================================
    # SAVE CHECKPOINT
    # =====================================================

    def save_checkpoint(self, epoch, best_acc, best=False):

        filename = f"{self.name}_best.pt" if best else f"{self.name}_latest.pt"

        path = os.path.join(self.ckpt_dir, filename)

        torch.save(
            {
                "epoch": epoch,
                "best_acc": best_acc,
                "model": self.model.state_dict(),
                "optimizer": self.optim.state_dict(),
                "scheduler": (
                    self.scheduler.state_dict() if self.scheduler is not None else None
                ),
                "scaler": self.scaler.state_dict(),
            },
            path,
        )

    # =====================================================
    # SAVE HISTORY
    # =====================================================

    def save_history(self, history):

        torch.save(history, os.path.join(self.ckpt_dir, "history.pt"))

    # =====================================================
    # TRAIN
    # =====================================================

    def train(self, epochs=5, start_epoch=0, best_acc=0.0):

        if self.scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optim, T_max=epochs, eta_min=1e-6
            )

        history = {
            "loss": [],
            "acc": [],
            "entropy": [],
            "diversity": [],
            "spread": [],
            "collapse_peak": [],
        }

        for epoch in range(start_epoch, epochs):

            printed_energy = False
            self.model.train()

            total_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in pbar:

                H = batch["H"].to(self.device)

                O = batch["O"].to(self.device)

                y = batch["y"].to(self.device)

                self.optim.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", enabled=(self.device == "cuda")):
                    outputs = self.model(H, O, y)
                    loss = compute_loss(outputs, y)

                if outputs.get("validator") is not None and not printed_energy:

                    energy = outputs["validator_potential"]

                    print(
                        f"\n[validator_potential] "
                        f"mean={energy.mean().item():.4f} "
                        f"max={energy.max().item():.4f} "
                        f"min={energy.min().item():.4f}"
                    )
                    print(
                        f"[Collapse Peak] "
                        f"{outputs['collapse_probs'].max(dim=1)[0].mean().item():.4f}"
                    )
                    entropy = (
                        -(
                            outputs["collapse_probs"]
                            * torch.log(outputs["collapse_probs"] + 1e-8)
                        )
                        .sum(dim=1)
                        .mean()
                    )
                    print(f"[Collapse Entropy] " f"{entropy.item():.4f}")

                    printed_energy = True

                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optim)

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    2.0,
                )

                self.scaler.step(self.optim)

                self.scaler.update()

                total_loss += loss.item()

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(self.train_loader)

            metrics = evaluate(self.model, self.val_loader, self.device)

            self.scheduler.step()
            print(f"LR: {self.scheduler.get_last_lr()[0]:.6f}")

            # ============================================
            # HISTORY
            # ============================================

            history["loss"].append(avg_loss)
            history["acc"].append(metrics["acc"])
            history["entropy"].append(metrics["entropy"])
            history["diversity"].append(metrics["diversity"])
            history["spread"].append(metrics["spread"])
            history["collapse_peak"].append(metrics["collapse_peak"])

            self.save_history(history)

            print("\n" + "=" * 60)
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss : " f"{avg_loss:.4f}")
            print(f"Val Acc    : " f"{metrics['acc']:.4f}")
            print(f"Entropy    : " f"{metrics['entropy']:.4f}")
            print(f"Diversity  : " f"{metrics['diversity']:.4f}")
            print(f"Spread     : " f"{metrics['spread']:.4f}")
            print(f"Collapse Peak : " f"{metrics['collapse_peak']:.4f}")

            if metrics["acc"] > best_acc:
                best_acc = metrics["acc"]
                self.save_checkpoint(epoch, best_acc, best=True)
                print(f"[BEST] " f"{best_acc:.4f}")

            self.save_checkpoint(epoch, best_acc, best=False)

        print("\nTraining Complete.")

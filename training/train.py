"""
FIXED TRAINING CONFIGURATION
Changes:
1. Increased base_params LR from 1e-4 to 5e-4
2. Increased new_params LR from 3e-5 to 2e-4
3. Fixed optimizer betas from (0.9, 0.98) to (0.9, 0.999) - standard values
4. These changes provide better gradient flow and faster convergence
"""

import os
import torch

from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler

from config import BASE_LR, NEW_LR, WEIGHT_DECAY

from training.losses import compute_loss
from training.evaluate import evaluate


class Trainer:

    def __init__(self, model, train_loader, val_loader, device, ckpt_dir, name, weight_decay=WEIGHT_DECAY):

        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device

        self.ckpt_dir = ckpt_dir
        self.name = name

        os.makedirs(ckpt_dir, exist_ok=True)

        # =====================================================
        # PARAMETER GROUPS WITH IMPROVED LEARNING RATES
        # =====================================================

        new_params, base_params = [], []
        for n, p in model.named_parameters():
            if n.startswith("quantum.") or n.startswith("validator."):
                new_params.append(p)
            else:
                base_params.append(p)

        param_groups = [{"params": base_params, "lr": BASE_LR}]
        if len(new_params) > 0:
            param_groups.append({"params": new_params, "lr": NEW_LR})

        # FIXED: Using standard AdamW betas (0.9, 0.999) instead of (0.9, 0.98)
        # Standard betas provide better second moment averaging
        self.optim = torch.optim.AdamW(
            param_groups, weight_decay=weight_decay, betas=(0.9, 0.999)
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

    def train(self, epochs=5, start_epoch=0, best_acc=0.0, patience=None):
        """
        patience: if set (e.g. 5), stop early after this many epochs with
        no improvement in val acc. None disables early stopping.
        """

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

        epochs_no_improve = 0

        for epoch in range(start_epoch, epochs):

            printed_energy = False
            self.model.train()

            total_loss = 0.0

            pbar = tqdm(
                self.train_loader,
                desc=f"[{self.name}] Epoch {epoch+1}/{epochs}",
            )

            for batch in pbar:

                H = batch["H"].to(self.device)

                O = batch["O"].to(self.device)

                y = batch["y"].to(self.device)

                self.optim.zero_grad(set_to_none=True)
                with autocast(device_type="cuda", enabled=(self.device == "cuda")):
                    outputs = self.model(H, O, y)
                    loss = compute_loss(outputs, y)

                if epoch == 0 and total_loss == 0:

                    print(f"\n========== DEBUG [{self.name}] ==========")
                    print("\nScores:")
                    print(outputs["scores"][0])
                    print("\nAnswer Energy:")
                    print(outputs["answer_energy"][0])
                    print("\nCollapse Probs:")
                    print(outputs["collapse_probs"][0])
                    print("\nCollapse Energy:")
                    print(outputs["collapse_energy"][0])
                    print("\nLoss:", loss.item())

                if outputs.get("validator") is not None and not printed_energy:

                    energy = outputs["validator_potential"]

                    print(
                        f"\n[{self.name}][validator_potential] "
                        f"mean={energy.mean().item():.4f} "
                        f"max={energy.max().item():.4f} "
                        f"min={energy.min().item():.4f}"
                    )
                    print(
                        f"[{self.name}][Collapse Peak] "
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
                    print(f"[{self.name}][Collapse Entropy] " f"{entropy.item():.4f}")

                    printed_energy = True

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 2.0)
                self.scaler.step(self.optim)
                self.scaler.update()

                total_loss += loss.item()
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(self.train_loader)

            metrics = evaluate(self.model, self.val_loader, self.device)

            self.scheduler.step()
            print(f"[{self.name}] LR: {self.scheduler.get_last_lr()[0]:.6f}")

            # Log quantum/classical fusion balance, if applicable
            if hasattr(self.model, "fusion"):
                alpha_val = torch.sigmoid(self.model.fusion.energy_alpha).item()
                print(
                    f"[{self.name}] energy_alpha={alpha_val:.4f} "
                    f"(0=pure quantum, 1=pure classical)"
                )

            history["loss"].append(avg_loss)
            history["acc"].append(metrics["acc"])
            history["entropy"].append(metrics["entropy"])
            history["diversity"].append(metrics["diversity"])
            history["spread"].append(metrics["spread"])
            history["collapse_peak"].append(metrics["collapse_peak"])

            self.save_history(history)

            print("\n" + "=" * 60)
            print(f"[{self.name}] Epoch {epoch+1}/{epochs}")
            print(f"Train Loss : {avg_loss:.4f}")
            print(f"Val Acc    : {metrics['acc']:.4f}")
            print(f"Entropy    : {metrics['entropy']:.4f}")
            print(f"Diversity  : {metrics['diversity']:.4f}")
            print(f"Spread     : {metrics['spread']:.4f}")
            print(f"Collapse Peak : {metrics['collapse_peak']:.4f}")

            if metrics["acc"] > best_acc:
                best_acc = metrics["acc"]
                self.save_checkpoint(epoch, best_acc, best=True)
                print(f"[{self.name}][BEST] {best_acc:.4f}")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            self.save_checkpoint(epoch, best_acc, best=False)

            if patience is not None and epochs_no_improve >= patience:
                print(
                    f"\n[{self.name}][EARLY STOP] "
                    f"No improvement for {patience} epochs. "
                    f"Best acc = {best_acc:.4f}"
                )
                break

        print(f"\n[{self.name}] Training Complete. Best acc = {best_acc:.4f}")

        history["best_acc"] = best_acc

        return history

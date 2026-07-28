import os
import torch

from tqdm.auto import tqdm
from torch.amp import autocast, GradScaler

from config import BASE_LR, NEW_LR, WEIGHT_DECAY

from training.losses import compute_loss
from training.evaluate import evaluate, batch_to_model


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        ckpt_dir,
        name,
        weight_decay=WEIGHT_DECAY,
        verbose=False,
    ):

        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device

        self.ckpt_dir = ckpt_dir
        self.name = name

        self.verbose = verbose

        os.makedirs(ckpt_dir, exist_ok=True)

        # =====================================================
        # PARAMETER GROUPS
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

        self.optim = torch.optim.AdamW(
            param_groups, weight_decay=weight_decay, betas=(0.9, 0.999)
        )

        # Same reasoning as the autocast device_type below: only pass
        # "cuda" when it's actually available, "cpu" otherwise (scaler
        # stays disabled either way on non-CUDA devices).
        self.scaler = GradScaler(
            "cuda" if device == "cuda" else "cpu",
            enabled=(device == "cuda"),
        )

        self.scheduler = None

    # =====================================================
    # PATHS -- per-run, never shared
    # =====================================================

    def history_path(self):
        """
        Every ablation config used to write ckpt_dir/history.pt with no
        run name, so seven configs clobbered one file and `load_or_resume`
        would hand a run some other run's curves. History is per-run now.
        """

        return os.path.join(self.ckpt_dir, f"{self.name}_history.pt")

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

        torch.save(history, self.history_path())

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
            "pairwise_cos": [],
            "h_cos": [],
            "interference_ratio": [],
            "phase_effect": [],
            "destructive_fraction": [],
            "ece": [],
        }

        epochs_no_improve = 0

        for epoch in range(start_epoch, epochs):

            self.model.train()

            total_loss = 0.0

            pbar = tqdm(
                self.train_loader,
                desc=f"[{self.name}] Epoch {epoch+1}/{epochs}",
            )

            for batch in pbar:

                inputs = batch_to_model(batch, self.device)

                y = batch["y"].to(self.device)

                self.optim.zero_grad(set_to_none=True)

                # device_type must match an actually-available backend even
                # when enabled=False -- "cuda" is only valid to pass when
                # self.device is "cuda"; "cpu" is a safe, always-valid
                # fallback for CPU/MPS runs (autocast stays disabled there).
                autocast_device_type = "cuda" if self.device == "cuda" else "cpu"

                with autocast(
                    device_type=autocast_device_type,
                    enabled=(self.device == "cuda"),
                ):
                    outputs = self.model(**inputs, y=y)
                    loss = compute_loss(outputs, y)

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

            history["loss"].append(avg_loss)
            for k in ("acc", "entropy", "diversity", "spread",
                      "collapse_peak", "pairwise_cos", "h_cos",
                      "interference_ratio", "phase_effect",
                      "destructive_fraction", "ece"):
                history[k].append(metrics[k])

            self.save_history(history)

            print("\n" + "=" * 60)
            print(f"[{self.name}] Epoch {epoch+1}/{epochs}   "
                  f"LR {self.scheduler.get_last_lr()[0]:.6f}")
            print(f"Train Loss    : {avg_loss:.4f}")
            print(f"Val Acc       : {metrics['acc']:.4f}")
            print(f"ECE           : {metrics['ece']:.4f}")
            print(f"Entropy       : {metrics['entropy']:.4f}")
            print(f"Diversity     : {metrics['diversity']:.6f}")
            print(f"Collapse Peak : {metrics['collapse_peak']:.4f}")
            print(f"Pairwise Cos  : {metrics['pairwise_cos']:.4f} (energy rows)")
            print(f"H Cos         : {metrics['h_cos']:.4f} (hypothesis vectors)")
            print(f"Interference  : {metrics['interference_ratio']:.4f} "
                  f"(|coherent - classical| / classical)")
            print(f"Phase effect  : {metrics['phase_effect']:.4f}   "
                  f"destructive {metrics['destructive_fraction']:.4f}")

            if self.verbose and hasattr(self.model, "fusion"):
                print(
                    f"energy_alpha={torch.sigmoid(self.model.fusion.energy_alpha).item():.4f} "
                    f"validator_alpha={torch.sigmoid(self.model.fusion.validator_alpha).item():.4f} "
                    f"dt={torch.sigmoid(self.model.reasoner.dt).item():.4f} "
                    f"anchor={torch.sigmoid(self.model.reasoner.anchor_weight).item():.4f} "
                    f"llm_prior={torch.nn.functional.softplus(self.model.llm_prior_weight).item():.4f}"
                )

            # The complex-amplitude machinery is the v45 claim, so it gets
            # the same treatment every other claim in this repo now gets:
            # a number, checked automatically, that says when it is inert.
            if getattr(self.model, "phase_mode", "classical") == "learned" and (
                metrics["phase_effect"] < 0.01
                or metrics["destructive_fraction"] < 0.01
            ):
                print(
                    f"[{self.name}][INTERFERENCE WARNING] "
                    f"phase_effect={metrics['phase_effect']:.5f}  "
                    f"destructive_fraction="
                    f"{metrics['destructive_fraction']:.5f}.\n"
                    f"  The phases are inert: the coherent sum is "
                    f"indistinguishable from the in-phase\n  one, and nothing "
                    f"is cancelling. Everything here is reproducible by a "
                    f"classical\n  mixture, so do not report a quantum result "
                    f"from this run -- the same failure the\n  v44 audit found, "
                    f"one level up."
                )

            # The check that would have caught the original failure.
            if metrics["pairwise_cos"] > 0.99:

                if metrics["h_cos"] > 0.99:
                    cause = (
                        "the REASONER has merged the hypotheses into one "
                        "vector (h_cos={:.4f}) -- an architecture problem"
                    ).format(metrics["h_cos"])
                else:
                    cause = (
                        "the hypotheses are still distinct (h_cos={:.4f}) but "
                        "the SELECTOR is ignoring them -- this is what a "
                        "low-signal cache looks like. Check the fallback rate "
                        "printed at load and regenerate; do not tune the model"
                    ).format(metrics["h_cos"])

                print(
                    f"[{self.name}][COLLAPSE WARNING] pairwise_cos="
                    f"{metrics['pairwise_cos']:.6f} -- the multi-hypothesis "
                    f"mechanism is inert and accuracy here reflects option "
                    f"priors only. Diagnosis: {cause}. Confirm with "
                    f"evaluation/input_ablation.py."
                )

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

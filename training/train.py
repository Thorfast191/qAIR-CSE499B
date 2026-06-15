import os
import torch

from tqdm.auto import tqdm

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

        self.optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # =====================================================
    # SAVE CHECKPOINT
    # =====================================================

    def save_checkpoint(self, epoch, best_acc,  best=False):

        filename = f"{self.name}_best.pt" if best else f"{self.name}_latest.pt"

        path = os.path.join(self.ckpt_dir, filename)

        torch.save(
            {
                "epoch": epoch,
                "best_acc": best_acc,
                "model": self.model.state_dict(),
                "optimizer": self.optim.state_dict(),
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

    def train(self, epochs=5, start_epoch=0):

        best_acc = 0.0

        best_path = os.path.join( self.ckpt_dir,  f"{self.name}_best.pt")

        if os.path.exists(best_path):

            try:

                best_ckpt = torch.load(
                    best_path,
                    map_location="cpu"
                )

                if "best_acc" in best_ckpt:

                    best_acc = best_ckpt[
                        "best_acc"
                    ]

                    print(
                        f"[BEST ACC RESTORED] "
                        f"{best_acc:.4f}"
                    )

            except Exception as e:
                print(
                    f"[WARNING] Could not load "
                    f"best_acc: {e}"
                )

        history = {"loss": [], "acc": [], "entropy": [], "diversity": [], "spread": []}

        for epoch in range(start_epoch,epochs):

            printed_energy = False
            self.model.train()

            total_loss = 0.0

            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{epochs}")

            for batch in pbar:

                H = batch["H"].to(self.device)

                O = batch["O"].to(self.device)

                y = batch["y"].to(self.device)

                outputs = self.model(H, O)
                if outputs["validator"] is not None and not printed_energy:

                    energy = outputs["validator"]["energy"]

                    print(
                        f"\n[Validator Energy] "
                        f"mean={energy.mean().item():.4f} "
                        f"max={energy.max().item():.4f} "
                        f"min={energy.min().item():.4f}"
                    )

                    printed_energy = True

                loss = compute_loss(outputs, y)

                self.optim.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optim.step()

                total_loss += loss.item()

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            avg_loss = total_loss / len(self.train_loader)

            metrics = evaluate(self.model, self.val_loader, self.device)

            # ============================================
            # HISTORY
            # ============================================

            history["loss"].append(avg_loss)

            history["acc"].append(metrics["acc"])

            history["entropy"].append(metrics["entropy"])

            history["diversity"].append(metrics["diversity"])

            history["spread"].append(metrics["spread"])

            self.save_history(history)

            print("\n" + "=" * 60)

            print(f"Epoch {epoch+1}/{epochs}")

            print(f"Train Loss : " f"{avg_loss:.4f}")

            print(f"Val Acc    : " f"{metrics['acc']:.4f}")

            print(f"Entropy    : " f"{metrics['entropy']:.4f}")

            print(f"Diversity  : " f"{metrics['diversity']:.4f}")

            print(f"Spread     : " f"{metrics['spread']:.4f}")

            self.save_checkpoint(epoch, best_acc, best=False)

            if metrics["acc"] > best_acc:

                best_acc = metrics["acc"]

                self.save_checkpoint(epoch, best_acc, best=True)

                print(f"[BEST] " f"{best_acc:.4f}")

        print("\nTraining Complete.")

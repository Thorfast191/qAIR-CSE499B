import os
import torch

from training.losses import compute_loss
from training.evaluate import evaluate


class Trainer:

    def __init__(

        self,

        model,

        train_loader,

        val_loader,

        device,

        ckpt_dir,

        meta_dir,

        name

    ):

        self.model = model

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.device = device

        self.ckpt_dir = ckpt_dir
        self.meta_dir = meta_dir

        self.name = name

        os.makedirs(ckpt_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)

        self.optim = torch.optim.AdamW(

            model.parameters(),

            lr=1e-4

        )

    # ========================================================
    # SAVE CHECKPOINT
    # ========================================================

    def save_ckpt(self, epoch):

        ckpt = {

            "model": self.model.state_dict(),

            "optim": self.optim.state_dict(),

            "epoch": epoch

        }

        # ----------------------------------------------------

        epoch_path = os.path.join(

            self.ckpt_dir,

            f"{self.name}_epoch{epoch}.pt"

        )

        torch.save(ckpt, epoch_path)

        # ----------------------------------------------------

        latest_path = os.path.join(

            self.ckpt_dir,

            f"{self.name}_latest.pt"

        )

        torch.save(ckpt, latest_path)

    # ========================================================
    # TRAIN
    # ========================================================

    def train(self, epochs=5):

        best_acc = 0.0

        for epoch in range(epochs):

            print("\n" + "=" * 60)
            print(f"Epoch {epoch + 1}/{epochs}")
            print("=" * 60)

            self.model.train()

            total_loss = 0.0

            # ------------------------------------------------
            # TRAIN LOOP
            # ------------------------------------------------

            for step, batch in enumerate(self.train_loader):

                H = batch["H"].to(self.device)
                O = batch["O"].to(self.device)
                y = batch["y"].to(self.device)

                outputs = self.model(H, O)

                loss = compute_loss(outputs, y)

                self.optim.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(

                    self.model.parameters(),

                    1.0

                )

                self.optim.step()

                total_loss += loss.item()

                # --------------------------------------------
                # LOGGING
                # --------------------------------------------

                if step % 10 == 0:

                    print(

                        f"Step {step:04d} | "
                        f"Loss = {loss.item():.4f}"

                    )

            # ------------------------------------------------
            # VALIDATION
            # ------------------------------------------------

            metrics = evaluate(

                self.model,

                self.val_loader,

                self.device

            )

            avg_loss = total_loss / len(self.train_loader)

            print("\nValidation Results")

            print(

                f"Loss     : {avg_loss:.4f}\n"
                f"Accuracy : {metrics['acc']:.4f}\n"
                f"Spread   : {metrics['spread']:.4f}"

            )

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            self.save_ckpt(epoch)

            # ------------------------------------------------
            # BEST MODEL
            # ------------------------------------------------

            if metrics["acc"] > best_acc:

                best_acc = metrics["acc"]

                best_path = os.path.join(

                    self.ckpt_dir,

                    f"{self.name}_best.pt"

                )

                torch.save(

                    self.model.state_dict(),

                    best_path

                )

                print(

                    f"[BEST] Saved: "
                    f"{best_path}"

                )

        print("\nTraining Complete.")
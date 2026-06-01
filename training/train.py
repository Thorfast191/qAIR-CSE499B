import os

import torch

from training.losses import (
    compute_loss
)

from training.evaluate import (
    evaluate
)


class Trainer:

    def __init__(

        self,

        model,

        train_loader,

        val_loader,

        device,

        ckpt_dir,

        name

    ):

        self.model = model

        self.train_loader = (
            train_loader
        )

        self.val_loader = (
            val_loader
        )

        self.device = device

        self.ckpt_dir = ckpt_dir

        self.name = name

        os.makedirs(
            ckpt_dir,
            exist_ok=True
        )

        self.optim = (
            torch.optim.AdamW(
                model.parameters(),
                lr=1e-4
            )
        )

    def train(
        self,
        epochs=5
    ):

        best_acc = 0.0

        for epoch in range(
            epochs
        ):

            self.model.train()

            total_loss = 0

            for batch in (
                self.train_loader
            ):

                H = batch["H"].to(
                    self.device
                )

                O = batch["O"].to(
                    self.device
                )

                y = batch["y"].to(
                    self.device
                )

                out = self.model(
                    H,
                    O
                )

                loss = compute_loss(
                    out,
                    y
                )

                self.optim.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0
                )

                self.optim.step()

                total_loss += (
                    loss.item()
                )

            metrics = evaluate(

                self.model,

                self.val_loader,

                self.device

            )

            print(
                f"\nEpoch {epoch+1}"
            )

            print(
                f"Loss={total_loss:.4f}"
            )

            print(
                f"Acc={metrics['acc']:.4f}"
            )

            print(
                f"Entropy={metrics['entropy']:.4f}"
            )

            print(
                f"Diversity={metrics['diversity']:.4f}"
            )

            ckpt_path = os.path.join(

                self.ckpt_dir,

                f"{self.name}_latest.pt"

            )

            torch.save(

                self.model.state_dict(),

                ckpt_path

            )

            if (
                metrics["acc"]
                > best_acc
            ):

                best_acc = (
                    metrics["acc"]
                )

                best_path = (
                    os.path.join(

                        self.ckpt_dir,

                        f"{self.name}_best.pt"

                    )
                )

                torch.save(

                    self.model.state_dict(),

                    best_path

                )
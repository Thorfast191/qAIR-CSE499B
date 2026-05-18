import os
            "epoch": epoch
        }, path)

        latest = os.path.join(
            self.ckpt_dir,
            f"{self.name}_latest.pt"
        )

        torch.save({
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "epoch": epoch
        }, latest)

    def train(self, epochs=10):

        best_acc = 0

        for epoch in range(epochs):

            self.model.train()

            for batch in self.train_loader:

                H = batch["H"].to(self.device)
                O = batch["O"].to(self.device)
                y = batch["y"].to(self.device)

                out = self.model(H, O)

                loss = compute_loss(out, y)

                self.optim.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    1.0
                )

                self.optim.step()

            metrics = evaluate(
                self.model,
                self.val_loader,
                self.device
            )

            print(
                f"Epoch {epoch+1} | "
                f"acc={metrics['acc']:.4f} | "
                f"spread={metrics['spread']:.4f}"
            )

            self.save_ckpt(epoch)

            if metrics["acc"] > best_acc:
                best_acc = metrics["acc"]

                torch.save(
                    self.model.state_dict(),
                    os.path.join(
                        self.ckpt_dir,
                        f"{self.name}_best.pt"
                    )
                )
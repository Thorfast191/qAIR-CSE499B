import torch
import torch.nn as nn
import torch.nn.functional as F


class CollapseController(nn.Module):

    def __init__(

        self,

        dim

    ):

        super().__init__()

        self.energy_head = nn.Linear(
            dim,
            1
        )

    def forward(self, H):

        energy = self.energy_head(
            H
        ).squeeze(-1)

        probs = F.softmax(
            -energy,
            dim=1
        )

        entropy = -(

            probs *

            torch.log(
                probs + 1e-8
            )

        ).sum(dim=1).mean()

        diversity = energy.var(
            dim=1
        ).mean()

        collapse_loss = (

            0.1 * entropy

            - diversity

        )

        return {

            "energy": energy,

            "probabilities": probs,

            "entropy": entropy,

            "diversity": diversity,

            "collapse_loss": collapse_loss

        }
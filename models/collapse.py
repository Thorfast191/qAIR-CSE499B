import torch
import torch.nn as nn


class CollapseController(nn.Module):

    def __init__(self):

        super().__init__()

    def forward(self, energy):

        amplitude = torch.exp(
            -0.5 * energy
        )

        amplitude = amplitude / (
            torch.norm(
                amplitude,
                dim=1,
                keepdim=True
            ) + 1e-8
        )

        probs = amplitude.pow(2)

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
            + 0.01 * diversity
        )

        return {

            "energy": energy,

            "amplitude": amplitude,

            "probabilities": probs,

            "entropy": entropy,

            "diversity": diversity,

            "collapse_loss": collapse_loss,
        }
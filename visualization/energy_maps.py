import matplotlib.pyplot as plt
import torch


def plot_energy_map(
    energies,
    save_path=None
):

    energies = torch.tensor(
        energies
    ).cpu().numpy()

    plt.figure(figsize=(8, 4))

    plt.imshow(
        energies,
        aspect="auto"
    )

    plt.colorbar(
        label="Energy"
    )

    plt.xlabel(
        "Hypothesis"
    )

    plt.ylabel(
        "Reasoning Step"
    )

    plt.title(
        "Hypothesis Energy Evolution"
    )

    if save_path:

        plt.savefig(
            save_path,
            bbox_inches="tight"
        )

    plt.show()
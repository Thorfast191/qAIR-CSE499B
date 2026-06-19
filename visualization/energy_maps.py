import matplotlib.pyplot as plt
import torch


def plot_energy_map(energies, save_path=None):

    if isinstance(energies, torch.Tensor):
        energies = energies.detach().cpu().numpy()

    plt.figure(figsize=(6,5))

    plt.imshow(
        energies,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(label="Hamiltonian Energy")

    plt.xlabel("Answer")

    plt.ylabel("Hypothesis")

    plt.title("Hypothesis ↔ Answer Energy")

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()
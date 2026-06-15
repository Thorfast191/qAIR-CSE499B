import matplotlib.pyplot as plt
import torch


def plot_attention_map(attention, save_path=None):

    if isinstance(attention, torch.Tensor):
        attention = attention.detach().cpu().numpy()

    plt.figure(figsize=(6, 6))

    plt.imshow(attention, aspect="auto")

    plt.colorbar()

    plt.xlabel("Hypothesis")

    plt.ylabel("Hypothesis")

    plt.title("Hypothesis Interaction Attention")

    if save_path:

        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_attention_map(attn_weights, title="Attention Interaction Map"):

    """
    attn_weights:
        shape = (K, K)
        or averaged attention matrix
    """

    attn_weights = np.array(attn_weights)

    plt.figure(figsize=(8, 6))

    sns.heatmap(
        attn_weights,
        annot=True,
        fmt=".2f",
        cmap="viridis"
    )

    plt.title(title)

    plt.xlabel("Target Hypothesis")
    plt.ylabel("Source Hypothesis")

    plt.show()
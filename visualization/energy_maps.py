import matplotlib.pyplot as plt
import numpy as np


def plot_energy_map(energy_matrix, title="Energy Landscape"):

    """
    energy_matrix:
        shape = (steps, hypotheses)
    """

    energy_matrix = np.array(energy_matrix)

    plt.figure(figsize=(10, 6))

    plt.imshow(
        energy_matrix,
        aspect='auto',
        interpolation='nearest'
    )

    plt.colorbar(label="Energy")

    plt.xlabel("Hypothesis")
    plt.ylabel("Reasoning Step")

    plt.title(title)

    plt.show()
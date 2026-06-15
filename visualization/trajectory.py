import torch
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA


def plot_trajectory(trajectory, save_path=None):

    vectors = []

    for step in trajectory:

        step = step[0].detach().cpu()

        vectors.append(step)

    vectors = torch.cat(vectors, dim=0)

    pca = PCA(n_components=2)

    coords = pca.fit_transform(vectors.numpy())

    plt.figure(figsize=(7, 7))

    plt.scatter(coords[:, 0], coords[:, 1])

    plt.plot(coords[:, 0], coords[:, 1])

    plt.xlabel("PC1")

    plt.ylabel("PC2")

    plt.title("Persistent Hypothesis Trajectory")

    if save_path:

        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

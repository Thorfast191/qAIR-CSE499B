import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np


def plot_trajectory(traj):

    X = []

    for t in traj:
        X.append(t.flatten())

    X = np.stack(X)

    pca = PCA(n_components=2)

    Z = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))

    plt.plot(Z[:, 0], Z[:, 1], marker="o")

    plt.title("Hypothesis Evolution Trajectory")

    plt.show()
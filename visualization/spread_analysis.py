import matplotlib.pyplot as plt


def plot_spread(spreads, save_path=None):

    plt.figure(figsize=(8, 4))

    plt.plot(spreads)

    plt.xlabel("Epoch")

    plt.ylabel("Spread")

    plt.title("Collapse Spread Evolution")

    plt.grid(True)

    if save_path:

        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

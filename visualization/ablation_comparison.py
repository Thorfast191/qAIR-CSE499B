import matplotlib.pyplot as plt


def plot_ablation_results(names, accuracies, save_path=None):

    plt.figure(figsize=(8, 5))

    plt.bar(names, accuracies)

    plt.title("Ablation Accuracy Comparison")

    plt.ylabel("Accuracy")

    plt.xticks(rotation=20)

    if save_path:

        plt.savefig(save_path, bbox_inches="tight")

    plt.show()

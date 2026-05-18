import matplotlib.pyplot as plt


def plot_spread(spreads):

    plt.figure(figsize=(8, 5))

    plt.plot(spreads)

    plt.title("Hypothesis Spread Evolution")

    plt.xlabel("Step")
    plt.ylabel("Spread")

    plt.show()
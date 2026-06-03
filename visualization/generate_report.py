import os
import torch
import pandas as pd
import matplotlib.pyplot as plt


def generate_report(
    history_path,
    export_dir
):

    os.makedirs(
        export_dir,
        exist_ok=True
    )

    # ====================================================
    # LOAD HISTORY
    # ====================================================

    history = torch.load(
        history_path,
        map_location="cpu"
    )

    epochs = list(
        range(
            1,
            len(history["acc"]) + 1
        )
    )

    # ====================================================
    # ACCURACY
    # ====================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["acc"],
        marker="o"
    )

    plt.title(
        "Validation Accuracy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Accuracy"
    )

    plt.grid(True)

    plt.savefig(
        os.path.join(
            export_dir,
            "accuracy.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    # ====================================================
    # ENTROPY
    # ====================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["entropy"],
        marker="o"
    )

    plt.title(
        "Collapse Entropy"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Entropy"
    )

    plt.grid(True)

    plt.savefig(
        os.path.join(
            export_dir,
            "entropy.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    # ====================================================
    # DIVERSITY
    # ====================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["diversity"],
        marker="o"
    )

    plt.title(
        "Hypothesis Diversity"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Diversity"
    )

    plt.grid(True)

    plt.savefig(
        os.path.join(
            export_dir,
            "diversity.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    # ====================================================
    # SPREAD
    # ====================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        epochs,
        history["spread"],
        marker="o"
    )

    plt.title(
        "Energy Spread"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Spread"
    )

    plt.grid(True)

    plt.savefig(
        os.path.join(
            export_dir,
            "spread.png"
        ),
        bbox_inches="tight"
    )

    plt.close()

    # ====================================================
    # CSV EXPORT
    # ====================================================

    df = pd.DataFrame({

        "epoch":
            epochs,

        "loss":
            history["loss"],

        "accuracy":
            history["acc"],

        "entropy":
            history["entropy"],

        "diversity":
            history["diversity"],

        "spread":
            history["spread"]

    })

    csv_path = os.path.join(
        export_dir,
        "metrics.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    # ====================================================
    # SUMMARY
    # ====================================================

    best_acc = max(
        history["acc"]
    )

    best_epoch = (
        history["acc"].index(
            best_acc
        )
        + 1
    )

    summary_path = os.path.join(
        export_dir,
        "summary.txt"
    )

    with open(
        summary_path,
        "w"
    ) as f:

        f.write(
            f"Best Accuracy: {best_acc:.4f}\n"
        )

        f.write(
            f"Best Epoch: {best_epoch}\n"
        )

        f.write(
            f"Final Loss: {history['loss'][-1]:.4f}\n"
        )

        f.write(
            f"Final Entropy: {history['entropy'][-1]:.4f}\n"
        )

        f.write(
            f"Final Diversity: {history['diversity'][-1]:.4f}\n"
        )

        f.write(
            f"Final Spread: {history['spread'][-1]:.4f}\n"
        )

    print(
        "[REPORT GENERATED]"
    )

    print(
        export_dir
    )
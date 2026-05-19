import argparse

from training.ablations import run_ablation_suite
from training.runner import run_training

# ============================================================
# ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(

        "--mode",

        type=str,

        default="train",

        choices=[
            "train",
            "ablation"
        ]
    )

    parser.add_argument(

        "--benchmark",

        type=str,

        default="arc"

    )

    parser.add_argument(

        "--epochs",

        type=int,

        default=5

    )

    parser.add_argument(

        "--samples",

        type=int,

        default=500

    )

    return parser.parse_args()

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    print("=" * 60)
    print("qAIR-vNext")
    print("=" * 60)

    print(f"Mode       : {args.mode}")
    print(f"Benchmark  : {args.benchmark}")
    print(f"Epochs     : {args.epochs}")
    print(f"Samples    : {args.samples}")

    # ========================================================
    # RUN
    # ========================================================

    if args.mode == "ablation":

        run_ablation_suite()

    elif args.mode == "train":

        run_training(

            benchmark=args.benchmark,

            epochs=args.epochs,

            max_samples=args.samples

        )
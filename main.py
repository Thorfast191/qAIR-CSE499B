import argparse

from config import CACHE_DIR, CKPT_DIR
from training.runner import run_training
from training.ablations import run_ablation_suite


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "ablation"]
    )

    args = parser.parse_args()

    if args.mode == "train":

        run_training()

    elif args.mode == "ablation":

        run_ablation_suite(cache_dir=CACHE_DIR, ckpt_dir=CKPT_DIR)


if __name__ == "__main__":

    main()

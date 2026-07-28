import argparse

from config import CACHE_DIR, CKPT_DIR, SEED, SEEDS, ARC_CONFIG
from training.seed import set_seed
from training.runner import run_training
from training.ablations import run_ablation_suite, ABLATIONS


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode", type=str, default="train", choices=["train", "ablation"]
    )

    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Seed for --mode train. Ablations sweep config.SEEDS.",
    )

    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(SEEDS),
        help="Seeds for --mode ablation (default: config.SEEDS).",
    )

    parser.add_argument(
        "--configs", type=str, nargs="+", default=None,
        choices=list(ABLATIONS),
        help="Subset of the ablation grid to run (default: all).",
    )

    parser.add_argument(
        "--arc-config", type=str, default=ARC_CONFIG,
        choices=["ARC-Challenge", "ARC-Easy", "merged"],
    )

    parser.add_argument(
        "--no-test", action="store_true",
        help="Skip held-out test evaluation (validation only).",
    )

    args = parser.parse_args()

    # set_seed lived in training/seed.py for the project's whole history
    # and was never called from anywhere, so no run was reproducible.
    set_seed(args.seed)

    if args.mode == "train":

        run_training(
            seed=args.seed,
            with_test=not args.no_test,
            arc_config=args.arc_config,
        )

    elif args.mode == "ablation":

        run_ablation_suite(
            cache_dir=CACHE_DIR,
            ckpt_dir=CKPT_DIR,
            seeds=tuple(args.seeds),
            with_test=not args.no_test,
            arc_config=args.arc_config,
            configs=args.configs,
        )


if __name__ == "__main__":

    main()

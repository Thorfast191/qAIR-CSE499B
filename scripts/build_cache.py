"""
Build the hypothesis/embedding cache for one or more ARC splits.

This is the slow, ONE-TIME step. Once `cache/arc_{split}.pt` exists,
every training run loads it instantly and never touches the LLM again.
You only need to rerun this if `config.EMBEDDING_DIM`/`EMBEDDING_MODEL`
changes (the dim guard in training/dataset.py will refuse a stale cache)
or if the generator changes.

Safe to interrupt. The build autosaves every 50 samples with the raw
dataset index it reached, and saves are atomic (temp file + os.replace),
so Ctrl-C, a crash, or a reboot costs you at most 50 samples -- rerun the
same command and it resumes where it stopped rather than restarting.

    python scripts/build_cache.py                    # train + validation
    python scripts/build_cache.py --splits test      # add the test split
    python scripts/build_cache.py --splits train validation test
    python scripts/build_cache.py --limit 32         # quick smoke test

On CPU expect roughly 10-11 s/question (~10 h for train's 3370, ~2.5 h
for validation's 869, ~10 h for test's ~3548). A GPU is 20-30x faster --
if you have access to one, build there and copy the .pt files over.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CACHE_DIR, resolve_device  # noqa: E402
from training.dataset import QAIRDataset  # noqa: E402


SPLIT_SIZES = {"train": 3370, "validation": 869, "test": 3548}


def main():

    parser = argparse.ArgumentParser(
        description="Build the ARC hypothesis/embedding cache (resumable).",
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "validation"],
        choices=["train", "validation", "test"],
    )
    parser.add_argument("--cache-dir", default=CACHE_DIR)
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Cap samples per split -- for a quick smoke test only.",
    )
    args = parser.parse_args()

    device = resolve_device()

    print("=" * 66)
    print("qAIR CACHE BUILD")
    print("=" * 66)
    print(f"device    : {device}")
    print(f"cache dir : {os.path.abspath(args.cache_dir)}")
    print(f"splits    : {', '.join(args.splits)}")

    if device == "cpu":
        est = sum(SPLIT_SIZES.get(s, 0) for s in args.splits) * 10.5 / 3600
        print(
            f"\n[CPU] Estimated ~{est:.1f} h total at ~10.5 s/question.\n"
            f"      Safe to interrupt -- rerun this exact command to resume.\n"
            f"      A GPU is 20-30x faster if you have one available."
        )

    print()

    started = time.time()

    for split in args.splits:

        print("\n" + "=" * 66)
        print(f"SPLIT: {split}")
        print("=" * 66)

        t0 = time.time()

        ds = QAIRDataset(
            split=split, max_samples=args.limit, cache_dir=args.cache_dir
        )

        print(
            f"[DONE {split}] {len(ds)} samples in "
            f"{(time.time() - t0) / 60:.1f} min"
        )

    print("\n" + "=" * 66)
    print(f"CACHE BUILD COMPLETE in {(time.time() - started) / 3600:.2f} h")
    print("=" * 66)
    print(
        "\nNext:\n"
        "  python main.py --mode train\n"
        "  python -m evaluation.input_ablation --ckpt ckpt/qair_v44_best.pt\n"
    )


if __name__ == "__main__":
    main()

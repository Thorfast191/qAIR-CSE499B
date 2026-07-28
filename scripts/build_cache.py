"""
Build the hypothesis/embedding cache for one or more ARC splits.

This is the slow, ONE-TIME step. Once `cache/arc_challenge_{split}.pt`
exists, every training run loads it instantly and never touches the LLM
again. Rerun only if `config.EMBEDDING_DIM`/`EMBEDDING_MODEL` changes
(the dim guard in training/dataset.py refuses a stale cache), if the
generator or its prompts change, or if `config.ARC_CONFIG` changes (a
different benchmark writes a different filename, so the two can't be
confused).

v45 does roughly TWICE the generation of v44 -- one supporting and one
attacking hypothesis per option -- plus a cheap scoring pass for the
per-option LLM log-likelihood. Budget accordingly.

Safe to interrupt. The build autosaves every 50 samples with the raw
dataset index it reached, and saves are atomic (temp file + os.replace),
so Ctrl-C, a crash, or a reboot costs you at most 50 samples -- rerun the
same command and it resumes where it stopped rather than restarting.

    python scripts/build_cache.py                       # train + validation
    python scripts/build_cache.py --splits test         # add the test split
    python scripts/build_cache.py --limit 80 20         # smoke-run caps
    python scripts/build_cache.py --arc-config merged

Progress goes to STDERR, because that is where tqdm writes. If you
redirect output, watch the stderr stream -- a quiet stdout does not mean
the build has stalled.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (  # noqa: E402
    CACHE_DIR,
    ARC_CONFIG,
    ARC_SPLIT_SIZES,
    GEN_MODES,
    resolve_device,
)
from training.dataset import QAIRDataset  # noqa: E402


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
        "--arc-config", default=ARC_CONFIG,
        choices=["ARC-Challenge", "ARC-Easy", "merged"],
    )
    parser.add_argument(
        "--limit", type=int, nargs="+", default=None,
        help=(
            "Cap samples per split. One value applies to every split; "
            "or give one value per split, in the same order as --splits. "
            "e.g. --splits train validation --limit 80 20"
        ),
    )
    args = parser.parse_args()

    sizes = ARC_SPLIT_SIZES.get(args.arc_config, {})

    if args.limit is None:
        limits = [None] * len(args.splits)
    elif len(args.limit) == 1:
        limits = args.limit * len(args.splits)
    elif len(args.limit) == len(args.splits):
        limits = args.limit
    else:
        parser.error(
            f"--limit takes 1 value or one per split; got {len(args.limit)} "
            f"for {len(args.splits)} splits."
        )

    device = resolve_device()

    print("=" * 66)
    print("qAIR CACHE BUILD")
    print("=" * 66)
    print(f"device    : {device}")
    print(f"benchmark : {args.arc_config}")
    print(f"cache dir : {os.path.abspath(args.cache_dir)}")
    print(f"splits    : {', '.join(args.splits)}")
    print(f"modes     : {', '.join(GEN_MODES)}  "
          f"({len(GEN_MODES)} hypotheses per option)")

    if device == "cpu":
        # ~7 s/question observed on 12 cores for v44's single-mode
        # generation; v45 generates len(GEN_MODES) times as many
        # completions per question. Do not take this figure from a short
        # run -- model loading (~40 s) dominates until a few hundred
        # samples in.
        n = sum(
            sizes.get(s, 0) if lim is None else min(lim, sizes.get(s, 0))
            for s, lim in zip(args.splits, limits)
        )
        per_q = 7.0 * len(GEN_MODES)
        print(
            f"\n[CPU] ~{n} questions, estimated ~{n * per_q / 3600:.1f} h "
            f"at ~{per_q:.0f} s/question.\n"
            f"      Safe to interrupt -- rerun this exact command to resume.\n"
            f"      A GPU is 20-30x faster; v45 is meant to be built on one."
        )

    print()

    started = time.time()

    for split, limit in zip(args.splits, limits):

        print("\n" + "=" * 66)
        print(f"SPLIT: {split}" + (f"  (limit {limit})" if limit else ""))
        print("=" * 66)

        t0 = time.time()

        ds = QAIRDataset(
            split=split, max_samples=limit, cache_dir=args.cache_dir,
            arc_config=args.arc_config,
        )

        print(
            f"[DONE {split}] {len(ds)} samples in "
            f"{(time.time() - t0) / 60:.1f} min"
        )

    print("\n" + "=" * 66)
    print(f"CACHE BUILD COMPLETE in {(time.time() - started) / 3600:.2f} h")
    print("=" * 66)
    print(
        "\nRead the [CACHE QUALITY] block above before training. If no "
        "hypothesis-derived\nstatistic beats chance, the prompts produced "
        "symmetric evidence and no amount of\ntraining will recover it -- fix "
        "models/generator.py and rebuild.\n"
        "\nNext:\n"
        "  python -m evaluation.baselines --mode all\n"
        "  python main.py --mode train\n"
        "  python -m evaluation.input_ablation --ckpt ckpt/qair_v45_best.pt\n"
    )


if __name__ == "__main__":
    main()

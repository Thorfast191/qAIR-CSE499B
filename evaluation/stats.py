"""
Statistical reporting for ablation comparisons.

ARC's validation split has 869 examples, so a single run's accuracy
carries a standard error of roughly

    sqrt(0.33 * 0.67 / 869) = 0.016

i.e. +/- 1.6 points at one sigma, +/- 3.2 at two. Most of the gaps this
project's ablation grid produces are smaller than that. Reporting a
single number per config -- as every result here did before the audit --
cannot distinguish an effect from seed noise.

Provides:
  * bootstrap_ci   -- accuracy with a percentile bootstrap CI
  * mcnemar        -- exact paired test between two arms
  * compare_arms   -- formatted table over ablation_results.pt

Paired tests matter here: two ablation arms see the SAME examples, so
McNemar on the disagreement cells is far more powerful than comparing
two independent proportions.
"""

import argparse
import math

import torch


def bootstrap_ci(records, n_boot=10000, alpha=0.05, seed=0):
    """
    records: iterable of 0/1 per-example correctness.
    Returns (accuracy, lo, hi) at the (1-alpha) level.
    """

    x = torch.tensor(records, dtype=torch.float)

    n = x.numel()

    g = torch.Generator().manual_seed(seed)

    idx = torch.randint(0, n, (n_boot, n), generator=g)

    boots = x[idx].mean(dim=1)

    lo = torch.quantile(boots, alpha / 2).item()
    hi = torch.quantile(boots, 1 - alpha / 2).item()

    return x.mean().item(), lo, hi


def mcnemar(records_a, records_b):
    """
    Exact McNemar test on paired per-example correctness.

    b = A right, B wrong;  c = A wrong, B right.
    Under H0 the split of the (b + c) discordant pairs is Binomial(n, 0.5).
    Returns (b, c, two_sided_p).
    """

    a = torch.tensor(records_a, dtype=torch.bool)
    b_ = torch.tensor(records_b, dtype=torch.bool)

    if a.numel() != b_.numel():
        raise ValueError(
            f"paired test needs equal-length records, got "
            f"{a.numel()} and {b_.numel()}"
        )

    b = int((a & ~b_).sum())
    c = int((~a & b_).sum())

    n = b + c

    if n == 0:
        return b, c, 1.0

    k = min(b, c)

    # Exact two-sided binomial tail.
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)

    return b, c, min(1.0, 2 * tail)


def compare_arms(results, baseline=None, alpha=0.05):
    """
    results: the dict saved by training/ablations.py.

    Pools per-example test records across seeds for each arm. Pooling is
    valid for the paired comparison because every seed of every arm is
    evaluated on the same test set in the same order.
    """

    pooled = {}

    for name, runs in results.items():
        recs = [r["test_records"] for r in runs if "test_records" in r]
        if recs:
            pooled[name] = [v for run in recs for v in run]

    if not pooled:
        print("No test_records found. Re-run the suite with with_test=True.")
        return

    if baseline is None:
        baseline = "A1_baseline" if "A1_baseline" in pooled else next(iter(pooled))

    print("\n" + "=" * 86)
    print(f"Accuracy with {int((1-alpha)*100)}% bootstrap CI, "
          f"and paired McNemar vs. {baseline}")
    print("=" * 86)
    print(f"{'config':<34s} {'acc':>7s} {'95% CI':>18s} "
          f"{'b/c':>11s} {'p':>9s}")

    base_recs = pooled[baseline]

    for name, recs in pooled.items():

        acc, lo, hi = bootstrap_ci(recs, alpha=alpha)

        if name == baseline:
            print(f"{name:<34s} {acc:>7.4f} [{lo:.4f}, {hi:.4f}] "
                  f"{'--':>11s} {'--':>9s}")
            continue

        b, c, p = mcnemar(base_recs, recs)

        star = " *" if p < alpha else ""

        print(f"{name:<34s} {acc:>7.4f} [{lo:.4f}, {hi:.4f}] "
              f"{f'{b}/{c}':>11s} {p:>9.4f}{star}")

    print("-" * 86)
    print("* = significant at the stated level. Overlapping CIs between two "
          "arms do NOT imply a non-significant difference -- read the paired "
          "p-value, not the intervals.")
    print("No multiple-comparison correction is applied. With 8 arms at "
          "alpha=0.05 you expect ~0.4 false positives; apply Holm-Bonferroni "
          "before claiming any single contrast in a paper.")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=str, default="ckpt/ablation_results.pt")
    parser.add_argument("--baseline", type=str, default=None)
    args = parser.parse_args()

    compare_arms(torch.load(args.results, map_location="cpu"), baseline=args.baseline)


if __name__ == "__main__":
    main()

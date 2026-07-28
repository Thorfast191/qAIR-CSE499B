"""
ARC loader.

v45: ARC-Challenge alone is the default, and merging is opt-in.

Every earlier version silently merged ARC-Challenge with ARC-Easy to
triple the pool (1,119 -> 3,370 train). Three problems with that, in
increasing order of seriousness:

  1. Published ARC numbers are always reported per config. A merged
     number is comparable to nothing in the literature.
  2. The two configs have very different difficulty distributions, so
     every measured effect is confounded with the mixture ratio -- and
     the ratio changes whenever either split is filtered.
  3. Nothing in the pipeline recorded WHICH benchmark a cached sample
     came from, so the mixture could not even be reported after the
     fact.

Each returned example now carries a `source` field ("ARC-Challenge" /
"ARC-Easy") so accuracy can be broken down per config even when the
merged pool is deliberately used for training volume.
"""

from datasets import load_dataset, concatenate_datasets, DatasetDict

from config import ARC_CONFIG

CONFIGS = ("ARC-Challenge", "ARC-Easy")


def _tagged(ds, name):
    """Attach the benchmark identity to every row."""

    return ds.map(lambda _: {"source": name})


def load_arc(arc_config=None):
    """
    arc_config: "ARC-Challenge" (default), "ARC-Easy", or "merged".
    """

    arc_config = arc_config or ARC_CONFIG

    if arc_config in CONFIGS:
        raw = load_dataset("ai2_arc", arc_config)
        return DatasetDict(
            {split: _tagged(raw[split], arc_config) for split in raw.keys()}
        )

    if arc_config != "merged":
        raise ValueError(
            f"arc_config must be one of {CONFIGS + ('merged',)}, "
            f"got {arc_config!r}"
        )

    challenge = load_dataset("ai2_arc", "ARC-Challenge")
    easy = load_dataset("ai2_arc", "ARC-Easy")

    merged = DatasetDict()

    for split in challenge.keys():

        parts = [_tagged(challenge[split], "ARC-Challenge")]

        if split in easy:
            parts.append(_tagged(easy[split], "ARC-Easy"))

        merged[split] = (
            concatenate_datasets(parts) if len(parts) > 1 else parts[0]
        )

    return merged

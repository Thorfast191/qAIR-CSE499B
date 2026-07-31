"""
Reference baselines. Run these BEFORE trusting any qAIR number.

The v44 audit found the full 8.2M-parameter model scoring 0.3383 on
validation while a 30-line MLP over option embeddings alone scored
0.3585 -- the architecture was net-NEGATIVE against its own simplest
baseline, and nothing in the repo would have revealed that.

Three baselines and one cache-level check, cheapest first:

  --mode llm      The generator's own per-option log-likelihood, read
                  straight out of the cache. Free, since v45 stores it at
                  build time. THIS IS THE NUMBER THAT DECIDES WHETHER THE
                  PROJECT HAS A RESULT: if the pipeline cannot beat the
                  model it is built on top of, it is a lossy compression
                  of that model's knowledge and no reviewer will look
                  past it. It went unmeasured for the project's entire
                  history.

  --mode asymmetry
                  Not a baseline -- a check on the premise the whole v45
                  design rests on: that the attacking hypothesis for the
                  CORRECT option is systematically weaker than the attacks
                  on the distractors. Free, cache-level, no training and
                  no GPU. If that asymmetry is absent there is nothing for
                  the coherent collapse to discriminate on, and no amount
                  of architecture work reaches it -- the fix belongs in
                  the attack prompt in models/generator.py.

  --mode probe    Trainable probes on the cached embeddings. Establishes
                  the DATA ceiling: no architecture over this cache can
                  do much better, so a qAIR number below these is a
                  problem with qAIR, not with the task.

                  Measured on the v44 cache (merged ARC, validation):
                      O only                      0.3585
                      diag(H_n, O_n)              0.3527
                      full pairwise H_k x O_n     0.3979
                      Q x O pairwise              0.4849   <- question!
                      Q x O + H                   0.4408   <- H HURT

  --mode direct   Re-scores the options with a freshly loaded LLM. Slow,
                  and should reproduce --mode llm exactly; kept as a
                  check that the cached scores are what they claim to be.

  --mode all      llm + asymmetry + probe.

    python -m evaluation.baselines --mode all --split validation
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from config import (
    CACHE_DIR,
    ARC_CONFIG,
    EMBEDDING_DIM as D,
    resolve_device,
)
from training.dataset import QAIRDataset
from training.seed import set_seed


# ==========================================================
# CACHED LLM BASELINE  (free)
# ==========================================================

def run_cached_llm_baseline(cache_dir, split="validation", max_samples=None,
                            arc_config=ARC_CONFIG):
    """
    argmax over the per-option log-likelihoods cached at build time.
    """

    ds = QAIRDataset(
        split=split, max_samples=max_samples, cache_dir=cache_dir,
        arc_config=arc_config,
    )

    correct = 0
    records = []

    for s in ds.samples:

        lp = s.get("llm_logprob")

        if lp is None or lp.numel() != len(s["options"]):
            continue

        hit = int(lp.argmax().item() == s["y"])

        correct += hit
        records.append(hit)

    acc = correct / max(len(records), 1)

    print(f"\n=== DIRECT LLM BASELINE (cached, {arc_config} {split}) ===")
    print(f"  length-normalized option likelihood : {acc:.4f}  "
          f"({correct}/{len(records)})")
    print(
        "\n  If this beats the trained qAIR pipeline, the architecture is a\n"
        "  lossy compression of knowledge the generator already had. That is\n"
        "  the first thing a reviewer will check."
    )

    return acc, records


# ==========================================================
# SUPPORT / ATTACK ASYMMETRY  (free, cache-level)
# ==========================================================

def run_asymmetry(cache_dir, split="validation", max_samples=None,
                  arc_config=ARC_CONFIG):
    """
    Measures the premise the v45 evidence design rests on: that the LLM
    objects to a genuinely wrong option more specifically than to a
    correct one, so the attacking channel carries an asymmetry the
    coherent collapse can discriminate on.

    The statistic is cos(H_k, O_{hyp_option[k]}) -- how closely each
    hypothesis tracks the option it was generated to argue about -- split
    by whether that option is the correct one. Supports are reported as
    the reference: a gap that appears in BOTH channels is a property of
    the options and the encoder, not of the attack prompt.

    Fallback rates come with it because a templated fallback is generic
    text carrying no evidence about any option, so if fallbacks
    concentrate in one group the cosine gap is partly measuring them.

    With the hypothesis phases at 0 and pi the coherent sum reduces to
    (sum_support w - sum_attack w)^2, so an attack gap indistinguishable
    from zero means the collapse has nothing to discriminate on -- and the
    fix is in the attack prompt in models/generator.py, not in the model.
    """

    ds = QAIRDataset(
        split=split, max_samples=max_samples, cache_dir=cache_dir,
        arc_config=arc_config,
    )

    # group -> is-the-correct-option -> per-hypothesis values
    cos = {"attack": {True: [], False: []}, "support": {True: [], False: []}}
    fell_back = {"attack": {True: [], False: []}, "support": {True: [], False: []}}

    n_samples = 0

    for s in ds.samples:

        pol = s.get("polarity")
        opt = s.get("hyp_option")

        if pol is None or opt is None:
            continue

        H = F.normalize(s["H"].float(), dim=-1)
        O = F.normalize(s["O"].float(), dim=-1)

        flags = s["is_fallback"]

        # Each hypothesis against the option it argues about, not against
        # every option -- alignment is positional in a v45 cache, so this
        # pairing is exactly the one generation used.
        d = (H * O[opt]).sum(-1)

        n_samples += 1

        for k in range(H.shape[0]):

            group = "attack" if pol[k].item() < 0 else "support"
            on_correct = bool(opt[k].item() == s["y"])

            cos[group][on_correct].append(d[k].item())
            fell_back[group][on_correct].append(float(flags[k].item()))

    def mean(xs):
        return sum(xs) / max(len(xs), 1)

    print(f"\n=== SUPPORT/ATTACK ASYMMETRY (cached, {arc_config} {split}) ===")
    print(f"  {n_samples} samples, split by whether the hypothesis argues "
          f"about the correct option")

    out = {}

    for key, header, table in (
        ("cos", "cos(H_k, O_hyp_option[k])", cos),
        ("fallback", "fallback rate", fell_back),
    ):

        print(f"\n  {header}")
        print(f"    {'':<9s}{'correct':>10s}{'distractor':>12s}{'gap':>10s}"
              f"{'n correct':>12s}{'n distractor':>14s}")

        for group in ("attack", "support"):

            hit = table[group][True]
            miss = table[group][False]

            c, w = mean(hit), mean(miss)

            print(f"    {group:<9s}{c:>10.4f}{w:>12.4f}{c - w:>+10.4f}"
                  f"{len(hit):>12d}{len(miss):>14d}")

            out[f"{group}_{key}_correct"] = c
            out[f"{group}_{key}_distractor"] = w
            out[f"{group}_{key}_gap"] = c - w

    print(
        "\n  The attack rows carry the claim; the support rows are the "
        "reference, since a\n  gap present in both is a property of the "
        "options rather than of the attack\n  prompt. With the hypothesis "
        "phases at 0 and pi the coherent sum reduces to\n  (sum_support w - "
        "sum_attack w)^2, so an attack gap indistinguishable from zero\n"
        "  means the coherent collapse has no amplitude asymmetry to work "
        "with."
    )

    return out


# ==========================================================
# PROBES ON CACHED EMBEDDINGS  (the data ceiling)
# ==========================================================

def _stack(ds, n_opt=4):
    """
    Returns (H_sup, H_att, O, Q, llm, y) for the samples with exactly
    n_opt options and the v45 two-channel hypothesis layout.
    """

    rows = [
        s for s in ds.samples
        if s["O"].shape[0] == n_opt
        and s["H"].shape[0] == 2 * n_opt
        and "Q" in s
    ]

    if not rows:
        raise RuntimeError(
            f"no {n_opt}-option samples with support+attack hypotheses found "
            f"-- is this a v45 cache?"
        )

    H = torch.stack([r["H"].float() for r in rows])

    return (
        H[:, :n_opt],
        H[:, n_opt:],
        torch.stack([r["O"].float() for r in rows]),
        torch.stack([r["Q"].float() for r in rows]),
        torch.stack([r["llm_logprob"].float() for r in rows]),
        torch.tensor([r["y"] for r in rows]),
    )


class Probe(nn.Module):

    def __init__(self, n_feat, n_scalar=0):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(n_feat * D + n_scalar, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1),
        )

    def forward(self, X):
        return self.f(X).squeeze(-1)


def _qexp(Q, O):
    return Q.unsqueeze(1).expand_as(O)


FEATURES = {
    "O only": (1, 0, lambda sup, att, O, Q, lp: O),

    "Q x O": (4, 0, lambda sup, att, O, Q, lp: torch.cat(
        [_qexp(Q, O), O, _qexp(Q, O) - O, _qexp(Q, O) * O], -1)),

    "Q x O + support": (6, 0, lambda sup, att, O, Q, lp: torch.cat(
        [_qexp(Q, O), O, _qexp(Q, O) - O, _qexp(Q, O) * O, sup, sup * O], -1)),

    "Q x O + support + attack": (8, 0, lambda sup, att, O, Q, lp: torch.cat(
        [_qexp(Q, O), O, _qexp(Q, O) - O, _qexp(Q, O) * O,
         sup, sup * O, att, att * O], -1)),

    "Q x O + evidence + llm": (8, 1, lambda sup, att, O, Q, lp: torch.cat(
        [_qexp(Q, O), O, _qexp(Q, O) - O, _qexp(Q, O) * O,
         sup, sup * O, att, att * O, _zscore(lp).unsqueeze(-1)], -1)),
}


def _zscore(x):
    return (x - x.mean(dim=1, keepdim=True)) / (x.std(dim=1, keepdim=True) + 1e-6)


def run_probes(cache_dir, epochs=60, lr=3e-3, train_samples=None,
               val_samples=None, arc_config=ARC_CONFIG):
    """
    train_samples/val_samples MUST match whatever cap the cache was built
    with. A cache built with max_samples is stored complete=False by
    design (so a later full build can resume and finish the split), which
    means calling QAIRDataset with max_samples=None against it does not
    read 80 samples -- it starts generating the rest of the split.
    """

    tr = _stack(QAIRDataset(
        split="train", max_samples=train_samples, cache_dir=cache_dir,
        arc_config=arc_config))
    va = _stack(QAIRDataset(
        split="validation", max_samples=val_samples, cache_dir=cache_dir,
        arc_config=arc_config))

    print(f"\ntrain={tr[5].numel()}  val={va[5].numel()}  chance=0.2500")
    print("\n=== DATA CEILING (probes on cached embeddings) ===")

    out = {}

    for name, (n_feat, n_scalar, fn) in FEATURES.items():

        set_seed(0)

        model = Probe(n_feat, n_scalar)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

        Xtr, ytr = fn(*tr[:5]), tr[5]
        Xva, yva = fn(*va[:5]), va[5]

        best = 0.0

        for _ in range(epochs):
            model.train()
            perm = torch.randperm(ytr.numel())
            for i in range(0, perm.numel(), 128):
                b = perm[i:i + 128]
                loss = F.cross_entropy(model(Xtr[b]), ytr[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                best = max(best, (model(Xva).argmax(1) == yva).float().mean().item())

        out[name] = best
        print(f"  {name:<28s} val acc = {best:.4f}")

    print(
        "\n  'Q x O + support' vs 'Q x O' is the question the whole project "
        "turns on:\n  does the generated evidence add anything to the "
        "question and the options?\n  On the v44 cache it was NEGATIVE "
        "(0.4849 -> 0.4408)."
    )

    return out


# ==========================================================
# LIVE LLM BASELINE  (verification of the cached scores)
# ==========================================================

@torch.no_grad()
def run_direct(cache_dir, split="validation", limit=None, max_samples=None,
               arc_config=ARC_CONFIG):
    """
    Scores each option by its mean token log-likelihood under the same
    LLM used for hypothesis generation, conditioned on the question.
    Length-normalized so long options aren't penalized.

    Should reproduce run_cached_llm_baseline; it exists so the cached
    numbers can be checked rather than trusted.
    """

    from models.generator import HypothesisGenerator, LLM_NAME

    gen = HypothesisGenerator(device=resolve_device())

    ds = QAIRDataset(
        split=split, max_samples=max_samples, cache_dir=cache_dir,
        arc_config=arc_config,
    )

    samples = ds.samples[:limit] if limit else ds.samples

    correct = 0
    records = []

    batch = 8

    for i in tqdm(range(0, len(samples), batch), desc=f"{LLM_NAME} direct"):

        chunk = samples[i:i + batch]

        scored = gen.score_options(
            [s["question"] for s in chunk],
            [s["options"] for s in chunk],
        )

        for s, scores in zip(chunk, scored):
            hit = int(max(range(len(scores)), key=lambda j: scores[j]) == s["y"])
            correct += hit
            records.append(hit)

    acc = correct / max(len(samples), 1)

    print(f"\n=== DIRECT LLM BASELINE (live, {LLM_NAME}, {split}) ===")
    print(f"  length-normalized option likelihood : {acc:.4f}  "
          f"({correct}/{len(samples)})")

    return acc, records


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["llm", "asymmetry", "probe", "direct", "all"],
        default="all",
    )
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR)
    parser.add_argument("--arc-config", type=str, default=ARC_CONFIG)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--train-samples", type=int, default=None,
        help="Must match the cap the train cache was built with.",
    )
    parser.add_argument(
        "--val-samples", type=int, default=None,
        help="Must match the cap the validation cache was built with.",
    )
    args = parser.parse_args()

    set_seed(0)

    max_for_split = (
        args.val_samples if args.split == "validation" else args.train_samples
    )

    if args.mode in ("llm", "all"):
        run_cached_llm_baseline(
            args.cache_dir, split=args.split, max_samples=max_for_split,
            arc_config=args.arc_config,
        )

    if args.mode in ("asymmetry", "all"):
        run_asymmetry(
            args.cache_dir, split=args.split, max_samples=max_for_split,
            arc_config=args.arc_config,
        )

    if args.mode in ("probe", "all"):
        run_probes(
            args.cache_dir,
            train_samples=args.train_samples,
            val_samples=args.val_samples,
            arc_config=args.arc_config,
        )

    if args.mode == "direct":
        run_direct(
            args.cache_dir, split=args.split, limit=args.limit,
            max_samples=max_for_split, arc_config=args.arc_config,
        )


if __name__ == "__main__":
    main()

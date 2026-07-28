import torch
import torch.nn.functional as F

from tqdm.auto import tqdm


def batch_to_model(batch, device):
    """
    One place that knows the model's input signature, so evaluate,
    input_ablation and the notebooks cannot drift on which fields get
    passed. Missing keys (a hand-built batch, a legacy cache) fall back
    to None rather than raising -- the model supplies defaults.
    """

    def get(name):
        v = batch.get(name)
        return v.to(device) if v is not None else None

    return {
        "H": get("H"),
        "O": get("O"),
        "Q": get("Q"),
        "H_mask": get("H_mask"),
        "O_mask": get("O_mask"),
        "polarity": get("polarity"),
        "align": get("align"),
        "llm_logprob": get("llm_logprob"),
    }


@torch.no_grad()
def evaluate(model, loader, device, return_records=False):
    """
    Accuracy plus the diagnostics whose absence let a total collapse of
    this project's central mechanism go unnoticed for its whole history.

    `pairwise_cos` is the important one: mean cosine similarity between
    distinct hypotheses' ENERGY PROFILES after reasoning. At 1.0 every
    hypothesis produced an identical row, the collapse distribution is
    uniform by construction, and the model is not doing multi-hypothesis
    reasoning whatever the accuracy says. The pre-v44 model sat at
    1.000000 while reporting 33.7%.

    v45 adds four:

      interference_ratio  mean|coherent - classical| / mean(classical).
                          Says the amplitudes are being summed before
                          squaring. Large even with zero phases, so it is
                          not on its own evidence of anything quantum.

      phase_effect        how far the learned phases move the result from
                          the in-phase sum. THIS is the number that says
                          whether the phase channel learned anything. At
                          ~0 the Born rule has degenerated back into a
                          softmax over a classical mixture and the
                          quantum claim fails, whatever the accuracy.

      destructive_fraction  share of (sample, option) pairs where the
                          cross-term is negative -- hypotheses actively
                          cancelling. Identically 0 for any real,
                          non-negative amplitude scheme, so a non-zero
                          value is the one measurement here that a
                          classical mixture cannot reproduce.

      ece                 15-bin expected calibration error. A model that
                          is right 34% of the time while reporting 90%
                          confidence is not usable even when its accuracy
                          is respectable, and nothing in this repo used
                          to measure that.

    return_records=True additionally returns per-example correctness,
    which evaluation/stats.py needs for bootstrap CIs and McNemar tests.
    """

    model.eval()

    correct = 0
    total = 0

    spreads, entropies, diversities = [], [], []
    collapse_peaks, pairwise_cos, h_cos = [], [], []
    interference_ratios, phase_effects, destructive = [], [], []

    conf_all, hit_all = [], []

    records = []

    for batch in tqdm(loader, desc="Validation", leave=False):

        inputs = batch_to_model(batch, device)

        y = batch["y"].to(device)

        out = model(**inputs)

        probs = out["scores"].exp()

        conf, pred = probs.max(dim=1)

        hit = (pred == y)

        correct += hit.sum().item()
        total += y.size(0)

        conf_all.append(conf.detach().cpu())
        hit_all.append(hit.detach().cpu())

        if return_records:
            records.extend(hit.detach().cpu().tolist())

        hypothesis_energy = out["answer_energy"].mean(dim=-1)  # (B, K)

        spreads.append(hypothesis_energy.var(dim=1).mean().item())

        entropies.append(out["entropy"].item())
        diversities.append(out["diversity"].item())

        collapse_peaks.append(out["collapse_probs"].max(dim=1)[0].mean().item())

        H_mask = inputs["H_mask"]

        pairwise_cos.append(mean_pairwise_cos(out, H_mask))
        h_cos.append(_pairwise(out["H_reasoned"], H_mask))

        interference_ratios.append(float(out["interference_ratio"]))
        phase_effects.append(float(out["phase_effect"]))
        destructive.append(float(out["destructive_fraction"]))

    conf_all = torch.cat(conf_all)
    hit_all = torch.cat(hit_all).float()

    metrics = {
        "acc": correct / total,
        "spread": sum(spreads) / len(spreads),
        "entropy": sum(entropies) / len(entropies),
        "diversity": sum(diversities) / len(diversities),
        "collapse_peak": sum(collapse_peaks) / len(collapse_peaks),
        "pairwise_cos": sum(pairwise_cos) / len(pairwise_cos),
        "h_cos": sum(h_cos) / len(h_cos),
        "interference_ratio": sum(interference_ratios) / len(interference_ratios),
        "phase_effect": sum(phase_effects) / len(phase_effects),
        "destructive_fraction": sum(destructive) / len(destructive),
        "ece": expected_calibration_error(conf_all, hit_all),
    }

    if return_records:
        return metrics, records

    return metrics


def expected_calibration_error(confidence, hits, n_bins=15):
    """
    Standard equal-width-bin ECE over the predicted class's probability.
    """

    ece = 0.0

    n = confidence.numel()

    if n == 0:
        return 0.0

    edges = torch.linspace(0.0, 1.0, n_bins + 1)

    for i in range(n_bins):

        lo, hi = edges[i], edges[i + 1]

        in_bin = (confidence > lo) & (confidence <= hi)

        count = int(in_bin.sum())

        if count == 0:
            continue

        acc = hits[in_bin].mean().item()
        conf = confidence[in_bin].mean().item()

        ece += (count / n) * abs(acc - conf)

    return ece


def _pairwise(X, mask, center=False):
    """
    Mean cosine similarity between DISTINCT rows of X (B, K, .),
    excluding padded rows and the diagonal.
    """

    if center:
        X = X - X.mean(dim=-1, keepdim=True)

    X = F.normalize(X, dim=-1)

    S = torch.einsum("bkd,bjd->bkj", X, X)

    K = S.shape[1]

    if mask is None:
        mask = torch.ones(S.shape[0], K, dtype=torch.bool, device=S.device)

    valid = mask.unsqueeze(2) & mask.unsqueeze(1)

    eye = torch.eye(K, dtype=torch.bool, device=S.device).unsqueeze(0)

    valid = valid & ~eye

    n = valid.sum(dim=(1, 2)).clamp(min=1)

    return ((S * valid).sum(dim=(1, 2)) / n).mean().item()


def mean_pairwise_cos(out, mask):
    """
    Collapse detector, measured on the per-hypothesis ENERGY PROFILE over
    options. 1.0 means every hypothesis produced an identical energy row,
    so the amplitudes are uniform and the marginalization degenerates to
    a plain mean -- the multi-hypothesis mechanism is inert.

    Read together with `h_cos` (the same statistic on the post-reasoning
    hypothesis vectors) this separates two different failures:

        h_cos ~ 1, pairwise_cos ~ 1  -> the REASONER collapsed the
                                        hypotheses into one vector
        h_cos < 1, pairwise_cos ~ 1  -> hypotheses stayed distinct but
                                        the SELECTOR learned to ignore
                                        them, which is what happens when
                                        H carries no usable signal.
                                        Check the cache-quality report;
                                        regenerate, don't tune.
    """

    return _pairwise(out["answer_energy"], mask, center=True)


@torch.no_grad()
def accuracy_by_source(model, dataset, loader, device):
    """
    ARC-Challenge and ARC-Easy have different difficulty distributions,
    and every published number reports them separately. When the merged
    pool is used for training volume, this breaks the reported accuracy
    back out per benchmark.

    Requires the loader to be unshuffled so record i corresponds to
    dataset.samples[i].
    """

    _, records = evaluate(model, loader, device, return_records=True)

    by_source = {}

    for sample, hit in zip(dataset.samples, records):
        key = sample.get("source", "unknown")
        stat = by_source.setdefault(key, [0, 0])
        stat[0] += int(hit)
        stat[1] += 1

    return {k: (c / n, n) for k, (c, n) in by_source.items()}

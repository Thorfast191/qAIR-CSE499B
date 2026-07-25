import torch
import torch.nn.functional as F

from tqdm.auto import tqdm


@torch.no_grad()
def evaluate(model, loader, device, return_records=False):
    """
    Adds the diagnostics whose absence let a total collapse of the
    model's central mechanism go unnoticed for the project's whole
    history.

    `pairwise_cos` is the important one: the mean cosine similarity
    between distinct hypotheses AFTER reasoning. If it approaches 1.0 the
    hypotheses have become the same vector, the collapse distribution is
    uniform by construction, and the model is not doing multi-hypothesis
    reasoning no matter what the accuracy says. The pre-fix model sat at
    1.000000 while reporting 33.7%.

    `collapse_peak` near 1/K and `diversity` near 0 are the same story
    seen from the collapse distribution.

    return_records=True additionally returns per-example correctness,
    which evaluation/stats.py needs for bootstrap CIs and McNemar tests.
    """

    model.eval()

    correct = 0
    total = 0

    spreads = []
    entropies = []
    diversities = []
    collapse_peaks = []
    pairwise_cos = []
    h_cos = []

    records = []

    for batch in tqdm(loader, desc="Validation", leave=False):

        H = batch["H"].to(device)
        O = batch["O"].to(device)
        Q = batch["Q"].to(device)
        y = batch["y"].to(device)
        H_mask = batch["H_mask"].to(device)
        O_mask = batch["O_mask"].to(device)

        out = model(H, O, Q=Q, H_mask=H_mask, O_mask=O_mask)

        pred = out["scores"].argmax(dim=1)

        hit = (pred == y)

        correct += hit.sum().item()
        total += y.size(0)

        if return_records:
            records.extend(hit.detach().cpu().tolist())

        # -----------------------------------------
        # Hypothesis energy spread
        # -----------------------------------------

        hypothesis_energy = out["answer_energy"].mean(dim=-1)  # (B, K)

        spreads.append(hypothesis_energy.var(dim=1).mean().item())

        # -----------------------------------------
        # Collapse statistics
        # -----------------------------------------

        entropies.append(out["entropy"].item())

        diversities.append(out["diversity"].item())

        collapse_peak = out["collapse_probs"].max(dim=1)[0].mean().item()

        collapse_peaks.append(collapse_peak)

        # -----------------------------------------
        # Hypothesis distinguishability (collapse detector)
        # -----------------------------------------

        pairwise_cos.append(mean_pairwise_cos(out, H_mask))

        h_cos.append(_pairwise(out["H_reasoned"], H_mask))

    metrics = {
        "acc": correct / total,
        "spread": sum(spreads) / len(spreads),
        "entropy": sum(entropies) / len(entropies),
        "diversity": sum(diversities) / len(diversities),
        "collapse_peak": sum(collapse_peaks) / len(collapse_peaks),
        "pairwise_cos": sum(pairwise_cos) / len(pairwise_cos),
        "h_cos": sum(h_cos) / len(h_cos),
    }

    if return_records:
        return metrics, records

    return metrics


def _pairwise(X, mask, center=False):
    """
    Mean cosine similarity between DISTINCT rows of X (B, K, ·),
    excluding padded rows and the diagonal.
    """

    if center:
        X = X - X.mean(dim=-1, keepdim=True)

    X = F.normalize(X, dim=-1)

    S = torch.einsum("bkd,bjd->bkj", X, X)

    K = S.shape[1]

    valid = mask.unsqueeze(2) & mask.unsqueeze(1)

    eye = torch.eye(K, dtype=torch.bool, device=S.device).unsqueeze(0)

    valid = valid & ~eye

    n = valid.sum(dim=(1, 2)).clamp(min=1)

    return ((S * valid).sum(dim=(1, 2)) / n).mean().item()


def mean_pairwise_cos(out, mask):
    """
    Collapse detector, measured on the per-hypothesis ENERGY PROFILE over
    options. 1.0 means every hypothesis produced an identical energy row,
    so `collapse_probs` is uniform and the marginalization degenerates to
    a plain mean -- the multi-hypothesis mechanism is inert.

    Read together with `h_cos` (the same statistic on the post-reasoning
    hypothesis vectors) this separates two different failures:

        h_cos ~ 1, pairwise_cos ~ 1  -> the REASONER collapsed the
                                        hypotheses into one vector
        h_cos < 1, pairwise_cos ~ 1  -> hypotheses stayed distinct but
                                        the SELECTOR learned to ignore
                                        them, which is what happens when
                                        H carries no usable signal
                                        (e.g. template-fallback
                                        hypotheses -- regenerate the
                                        cache, don't touch the model)
    """

    return _pairwise(out["answer_energy"], mask, center=True)

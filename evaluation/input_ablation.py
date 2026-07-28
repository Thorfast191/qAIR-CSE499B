"""
Input-ablation regression test.

This is the single most important test in the project, and its absence
is why a total failure of the central mechanism went unnoticed while the
model reported a plausible-looking 33.7%.

qAIR claims to reason over MULTIPLE HYPOTHESES held in superposition.
That claim is falsifiable in one line: corrupt the hypotheses and see
whether the answer changes. On the pre-audit model it did not -- not
"changed a little", but bit-identically unchanged at 294/869 under every
corruption tried:

    intact H, O                        294/869 = 0.3383
    H := zeros                         294/869 = 0.3383
    H := gaussian noise                294/869 = 0.3383
    H rows shuffled within sample      294/869 = 0.3383
    H from a DIFFERENT question        294/869 = 0.3383
    H := O (pure option echo)          294/869 = 0.3383
    O := zeros                         223/869 = 0.2566   <- only O mattered

v45 makes this per-channel, because the model now has more inputs and
"the hypotheses matter" is no longer the only question worth asking. In
particular the cached LLM log-prior is a strong feature that can carry
the accuracy by itself -- a model that scores well with `H := zeros` and
collapses with `llm_logprob := zeros` is a wrapper around its own
generator, and reporting it as multi-hypothesis reasoning would be the
same category of error the v44 audit found, one level up.

    python -m evaluation.input_ablation --ckpt ckpt/qair_v45_best.pt
"""

import argparse
import os

import torch
from functools import partial
from torch.utils.data import DataLoader

from config import (
    CACHE_DIR,
    BATCH_SIZE,
    EMBEDDING_DIM,
    MODEL_DIM,
    N_QUBITS,
    ARC_CONFIG,
    resolve_device,
)
from training.dataset import QAIRDataset, collate_fn
from training.evaluate import batch_to_model
from training.seed import set_seed
from models.full_model import QAIRvNext


def _zero_where(inputs, keep):
    """Zero the hypothesis rows where `keep` is False."""

    H = inputs["H"].clone()

    H[~keep] = 0.0

    return {**inputs, "H": H}


CORRUPTIONS = {

    "intact": lambda x: x,

    # ---- the whole hypothesis channel ----

    "H := zeros": lambda x: {**x, "H": torch.zeros_like(x["H"])},

    "H := noise": lambda x: {**x, "H": torch.randn_like(x["H"]) * x["H"].std()},

    "H shuffled within sample": lambda x: {
        **x, "H": x["H"][:, torch.randperm(x["H"].shape[1], device=x["H"].device)]
    },

    "H rolled across batch": lambda x: {**x, "H": torch.roll(x["H"], 1, dims=0)},

    "H := O (option echo)": lambda x: {
        **x,
        "H": x["O"].repeat(1, (x["H"].shape[1] + x["O"].shape[1] - 1)
                           // x["O"].shape[1], 1)[:, : x["H"].shape[1]].clone(),
    },

    # ---- one evidence direction at a time ----

    "support := zeros": lambda x: _zero_where(x, x["polarity"] < 0),

    "attack := zeros": lambda x: _zero_where(x, x["polarity"] > 0),

    # ---- the structural tags ----

    "polarity := 0": lambda x: {**x, "polarity": torch.zeros_like(x["polarity"])},

    "align := 0": lambda x: {**x, "align": torch.zeros_like(x["align"])},

    # ---- the other inputs ----

    "llm_logprob := zeros": lambda x: {
        **x, "llm_logprob": torch.zeros_like(x["llm_logprob"])
    },

    "O := zeros": lambda x: {**x, "O": torch.zeros_like(x["O"])},
}

HYPOTHESIS_CHANNELS = (
    "H := zeros",
    "H := noise",
    "H shuffled within sample",
    "H rolled across batch",
    "H := O (option echo)",
    "support := zeros",
    "attack := zeros",
)


@torch.no_grad()
def input_ablation(model, loader, device):

    model.eval()

    results = {}

    for name, fn in CORRUPTIONS.items():

        correct = 0
        total = 0

        for batch in loader:

            inputs = batch_to_model(batch, device)

            y = batch["y"].to(device)

            out = model(**fn(inputs))

            correct += (out["scores"].argmax(dim=1) == y).sum().item()
            total += y.size(0)

        results[name] = correct / total

    return results


def report(results, threshold=0.005):

    base = results["intact"]

    print("\n" + "=" * 72)
    print("INPUT ABLATION -- which inputs is this model actually using?")
    print("=" * 72)

    for name, acc in results.items():

        delta = acc - base

        flag = ""

        if name in HYPOTHESIS_CHANNELS and abs(delta) < 1e-9:
            flag = "   <-- IDENTICAL: channel unused"

        print(f"  {name:<28s} {acc:.4f}   delta={delta:+.4f}{flag}")

    h_deltas = [results[k] - base for k in HYPOTHESIS_CHANNELS if k in results]

    worst = min(h_deltas) if h_deltas else 0.0

    llm_delta = results.get("llm_logprob := zeros", base) - base

    print("-" * 72)

    passed = worst <= -threshold

    if not passed:
        print(
            f"VERDICT: FAIL. Corrupting the hypotheses costs at most "
            f"{-worst:.4f} accuracy.\n"
            f"  The multi-hypothesis mechanism is inert; this model is an "
            f"option-prior\n  classifier. Do not report quantum, collapse, or "
            f"superposition results from\n  this checkpoint."
        )
    else:
        print(
            f"VERDICT: PASS. Destroying the hypotheses costs {-worst:.4f} "
            f"accuracy, so the\n  model is genuinely using them."
        )

    print()

    if llm_delta > -threshold:
        print(
            f"  LLM prior      : removing it costs {-llm_delta:.4f} -- the "
            f"cached generator\n                   likelihood is not carrying "
            f"this result."
        )
    elif -llm_delta > -worst:
        print(
            f"  LLM prior      : removing it costs {-llm_delta:.4f}, MORE than "
            f"destroying the\n                   hypotheses ({-worst:.4f}). "
            f"Most of this accuracy is the generator's,\n                   "
            f"not the architecture's. Report the use_llm_prior=False arm "
            f"as the\n                   reasoning result."
        )
    else:
        print(
            f"  LLM prior      : removing it costs {-llm_delta:.4f}; "
            f"hypotheses cost {-worst:.4f}.\n                   Both channels "
            f"contribute."
        )

    print("=" * 72)

    return passed


def build_model_from_config(cfg, device):

    return QAIRvNext(
        dim=EMBEDDING_DIM,
        model_dim=cfg.get("model_dim", MODEL_DIM),
        use_quantum=cfg.get("use_quantum", False),
        use_validator=cfg.get("use_validator", False),
        persistent_steps=cfg.get("persistent_steps", 3),
        n_qubits=cfg.get("n_qubits", N_QUBITS),
        backend=cfg.get("backend"),
        use_question=cfg.get("use_question", True),
        phase_mode=cfg.get("phase_mode", "learned"),
        phase_source=cfg.get("phase_source", "circuit"),
        mixing=cfg.get("mixing", "orthogonal"),
        use_llm_prior=cfg.get("use_llm_prior", True),
        use_attack=cfg.get("use_attack", True),
    ).to(device)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR)
    parser.add_argument("--arc-config", type=str, default=ARC_CONFIG)
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    set_seed(0)

    device = resolve_device()

    cfg_path = args.config or args.ckpt.replace("_best.pt", "_config.pt")

    if os.path.exists(cfg_path):
        cfg = torch.load(cfg_path, map_location="cpu")
    else:
        print(f"[WARN] no config at {cfg_path}; assuming full hybrid")
        cfg = {"use_quantum": True, "use_validator": True, "persistent_steps": 3}

    model = build_model_from_config(cfg, device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])

    ds = QAIRDataset(
        split=args.split, max_samples=args.max_samples,
        cache_dir=args.cache_dir, arc_config=args.arc_config,
    )

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, shuffle_options=False),
    )

    passed = report(input_ablation(model, loader, device))

    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()

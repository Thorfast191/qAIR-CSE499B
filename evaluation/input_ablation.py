"""
Input-ablation regression test.

This is the single most important test in the project, and its absence
is why a total failure of the central mechanism went unnoticed while the
model reported a plausible-looking 33.7%.

The claim qAIR makes is that it reasons over MULTIPLE HYPOTHESES held in
superposition. That claim is falsifiable in one line: corrupt the
hypotheses and see whether the answer changes. On the pre-audit model it
did not -- not "changed a little", but bit-identically unchanged at
294/869 across every corruption tried:

    intact H, O                        294/869 = 0.3383
    H := zeros                         294/869 = 0.3383
    H := gaussian noise                294/869 = 0.3383
    H rows shuffled within sample      294/869 = 0.3383
    H from a DIFFERENT question        294/869 = 0.3383
    H := O (pure option echo)          294/869 = 0.3383
    O := zeros                         223/869 = 0.2566   <- only O mattered

Run this after every training run. If `H := zeros` does not COST
accuracy, the model is not using hypotheses and any quantum/collapse
result you report is about something else.

    python -m evaluation.input_ablation --ckpt ckpt/qair_v44_best.pt
"""

import argparse
import os

import torch
from functools import partial
from torch.utils.data import DataLoader

from config import CACHE_DIR, CKPT_DIR, BATCH_SIZE, EMBEDDING_DIM, N_QUBITS, resolve_device
from training.dataset import QAIRDataset, collate_fn
from training.seed import set_seed
from models.full_model import QAIRvNext


CORRUPTIONS = {
    "intact": lambda H, O: (H, O),
    "H := zeros": lambda H, O: (torch.zeros_like(H), O),
    "H := noise": lambda H, O: (torch.randn_like(H) * H.std(), O),
    "H shuffled within sample": lambda H, O: (
        H[:, torch.randperm(H.shape[1], device=H.device)], O
    ),
    "H rolled across batch": lambda H, O: (torch.roll(H, 1, dims=0), O),
    "H := O": lambda H, O: (O[:, : H.shape[1]].clone(), O),
    "O := zeros": lambda H, O: (H, torch.zeros_like(O)),
}


@torch.no_grad()
def input_ablation(model, loader, device):

    model.eval()

    results = {}

    for name, fn in CORRUPTIONS.items():

        correct = 0
        total = 0

        for batch in loader:

            H = batch["H"].to(device)
            O = batch["O"].to(device)
            Q = batch["Q"].to(device)
            y = batch["y"].to(device)
            H_mask = batch["H_mask"].to(device)
            O_mask = batch["O_mask"].to(device)

            H2, O2 = fn(H, O)

            out = model(H2, O2, Q=Q, H_mask=H_mask, O_mask=O_mask)

            correct += (out["scores"].argmax(dim=1) == y).sum().item()
            total += y.size(0)

        results[name] = correct / total

    return results


def report(results):

    base = results["intact"]

    print("\n" + "=" * 68)
    print("INPUT ABLATION -- does the model actually use its hypotheses?")
    print("=" * 68)

    for name, acc in results.items():
        delta = acc - base
        flag = ""
        if name.startswith("H") and name != "intact" and abs(delta) < 1e-9:
            flag = "   <-- IDENTICAL: H is unused"
        print(f"  {name:<28s} {acc:.4f}   delta={delta:+.4f}{flag}")

    h_deltas = [
        results[k] - base
        for k in results
        if k.startswith("H") and k != "intact"
    ]

    worst = min(h_deltas) if h_deltas else 0.0

    print("-" * 68)

    if worst > -0.005:
        print(
            "VERDICT: FAIL. Corrupting the hypotheses costs at most "
            f"{-worst:.4f} accuracy. The multi-hypothesis mechanism is inert; "
            "this model is an option-prior classifier. Do not report quantum, "
            "collapse, or superposition results from this checkpoint."
        )
    else:
        print(
            f"VERDICT: PASS. Destroying the hypotheses costs {-worst:.4f} "
            "accuracy, so the model is genuinely using them."
        )

    print("=" * 68)

    return worst <= -0.005


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR)
    args = parser.parse_args()

    set_seed(0)

    device = resolve_device()

    cfg_path = args.config or args.ckpt.replace("_best.pt", "_config.pt")

    if os.path.exists(cfg_path):
        cfg = torch.load(cfg_path, map_location="cpu")
    else:
        print(f"[WARN] no config at {cfg_path}; assuming full hybrid")
        cfg = {
            "use_quantum": True,
            "use_validator": True,
            "persistent_steps": 3,
            "n_qubits": N_QUBITS,
        }

    model = QAIRvNext(
        dim=EMBEDDING_DIM,
        use_quantum=cfg.get("use_quantum", False),
        use_validator=cfg["use_validator"],
        persistent_steps=cfg["persistent_steps"],
        n_qubits=cfg.get("n_qubits", N_QUBITS),
        backend=cfg.get("backend"),
        use_question=cfg.get("use_question", True),
    ).to(device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device)["model"])

    ds = QAIRDataset(split=args.split, cache_dir=args.cache_dir)

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

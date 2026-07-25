"""
Reference baselines. Run these BEFORE trusting any qAIR number.

The audit found the full 8.2M-parameter model scoring 0.3383 on
validation while a 30-line MLP over option embeddings alone scored
0.3585 -- the architecture was net-NEGATIVE against its own simplest
baseline, and nothing in the repo would have revealed that.

Three baselines, cheapest first:

  --mode probe    Trainable probes on the cached embeddings. Establishes
                  the DATA ceiling: no architecture over this cache can
                  do much better, so a qAIR number below these is a
                  problem with qAIR, not with the task.

                  Measured pre-fix (validation):
                      O only                      0.3585
                      diag(H_n, O_n)              0.3527
                      full pairwise H_k x O_n     0.3979
                      Q x O pairwise              0.4849   <- question!
                      Q x O + H                   0.4408   <- H HURT

  --mode direct   Ask the generator LLM (Qwen2.5-0.5B-Instruct) the
                  question directly by scoring each option's likelihood.
                  This is the baseline that decides whether the project
                  has a result at all: if the pipeline cannot beat the
                  model it is built on top of, it is a lossy compression
                  of that model's knowledge, and no reviewer will look
                  past it.

  --mode all      Both.

    python -m evaluation.baselines --mode all --split validation
"""

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from config import CACHE_DIR, EMBEDDING_DIM as D, resolve_device
from training.dataset import QAIRDataset
from training.seed import set_seed


# ==========================================================
# PROBES ON CACHED EMBEDDINGS  (the data ceiling)
# ==========================================================

def _stack(ds, n_opt=4):

    rows = [
        s for s in ds.samples
        if s["H"].shape[0] == n_opt and s["O"].shape[0] == n_opt and "Q" in s
    ]

    return (
        torch.stack([r["H"].float() for r in rows]),
        torch.stack([r["O"].float() for r in rows]),
        torch.stack([r["Q"].float() for r in rows]),
        torch.tensor([r["y"] for r in rows]),
    )


class Probe(nn.Module):

    def __init__(self, n_feat):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(n_feat * D, 512), nn.GELU(), nn.Dropout(0.2), nn.Linear(512, 1)
        )

    def forward(self, X):
        return self.f(X).squeeze(-1)


FEATURES = {
    "O only": (1, lambda H, O, Q: O),
    "diag(H_n, O_n)": (4, lambda H, O, Q: torch.cat([H, O, H - O, H * O], -1)),
    "Q x O": (4, lambda H, O, Q: torch.cat(
        [Q.unsqueeze(1).expand_as(O), O,
         Q.unsqueeze(1).expand_as(O) - O, Q.unsqueeze(1).expand_as(O) * O], -1)),
    "Q x O + H": (6, lambda H, O, Q: torch.cat(
        [Q.unsqueeze(1).expand_as(O), O,
         Q.unsqueeze(1).expand_as(O) - O, Q.unsqueeze(1).expand_as(O) * O,
         H, H * O], -1)),
}


def run_probes(cache_dir, epochs=60, lr=3e-3):

    tr = _stack(QAIRDataset(split="train", cache_dir=cache_dir))
    va = _stack(QAIRDataset(split="validation", cache_dir=cache_dir))

    print(f"\ntrain={tr[3].numel()}  val={va[3].numel()}  chance=0.2500")
    print("\n=== DATA CEILING (probes on cached embeddings) ===")

    out = {}

    for name, (n_feat, fn) in FEATURES.items():

        set_seed(0)

        model = Probe(n_feat)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

        Xtr, ytr = fn(*tr[:3]), tr[3]
        Xva, yva = fn(*va[:3]), va[3]

        best = 0.0

        for _ in range(epochs):
            model.train()
            perm = torch.randperm(ytr.numel())
            for i in range(0, perm.numel(), 128):
                b = perm[i:i + 128]
                loss = F.cross_entropy(model(Xtr[b]), ytr[b])
                opt.zero_grad(); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                best = max(best, (model(Xva).argmax(1) == yva).float().mean().item())

        out[name] = best
        print(f"  {name:<28s} val acc = {best:.4f}")

    return out


# ==========================================================
# DIRECT LLM BASELINE
# ==========================================================

@torch.no_grad()
def run_direct(cache_dir, split="validation", limit=None):
    """
    Scores each option by its mean token log-likelihood under the same
    LLM used for hypothesis generation, conditioned on the question.
    Length-normalized so long options aren't penalized.
    """

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from models.generator import LLM_NAME

    device = resolve_device()

    tok = AutoTokenizer.from_pretrained(LLM_NAME, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.float16 if device == "cuda" else torch.float32
    lm = AutoModelForCausalLM.from_pretrained(
        LLM_NAME, torch_dtype=dtype, trust_remote_code=True
    ).to(device)
    lm.eval()

    ds = QAIRDataset(split=split, cache_dir=cache_dir)

    samples = ds.samples[:limit] if limit else ds.samples

    correct = 0
    records = []

    for s in tqdm(samples, desc=f"{LLM_NAME} direct"):

        scores = []

        for opt in s["options"]:

            prompt = f"Question: {s['question']}\nAnswer:"
            full = prompt + " " + opt

            p_ids = tok(prompt, return_tensors="pt").input_ids.to(device)
            f_ids = tok(full, return_tensors="pt").input_ids.to(device)

            logits = lm(f_ids).logits[:, :-1]
            targets = f_ids[:, 1:]

            logprobs = torch.log_softmax(logits.float(), dim=-1).gather(
                2, targets.unsqueeze(-1)
            ).squeeze(-1)

            n_prompt = p_ids.shape[1] - 1
            answer_lp = logprobs[0, n_prompt:]

            scores.append(answer_lp.mean().item() if answer_lp.numel() else -1e9)

        hit = int(max(range(len(scores)), key=lambda i: scores[i]) == s["y"])
        correct += hit
        records.append(hit)

    acc = correct / len(samples)

    print(f"\n=== DIRECT LLM BASELINE ({LLM_NAME}, {split}) ===")
    print(f"  length-normalized option likelihood : {acc:.4f}  "
          f"({correct}/{len(samples)})")
    print(
        "\n  If this beats the trained qAIR pipeline, the architecture is a\n"
        "  lossy compression of knowledge the generator already had. That is\n"
        "  the first thing a reviewer will check."
    )

    return acc, records


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["probe", "direct", "all"], default="all")
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--cache-dir", type=str, default=CACHE_DIR)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    set_seed(0)

    if args.mode in ("probe", "all"):
        run_probes(args.cache_dir)

    if args.mode in ("direct", "all"):
        run_direct(args.cache_dir, split=args.split, limit=args.limit)


if __name__ == "__main__":
    main()

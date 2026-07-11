import os
import time
import torch

from tqdm.auto import tqdm
from torch.utils.data import Dataset

from benchmarks.arc import load_arc

from models.generator import HypothesisGenerator
from models.encoder import HypothesisEncoder

DIM = 384


class QAIRDataset(Dataset):

    def __init__(self, split="train", max_samples=None, cache_dir="./cache"):

        self.samples = []

        os.makedirs(cache_dir, exist_ok=True)

        cache_path = os.path.join(cache_dir, f"arc_{split}.pt")

        # ====================================================
        # LOAD CACHE
        # ====================================================

        if os.path.exists(cache_path):

            print(f"[CACHE FOUND] {cache_path}")

            loaded = torch.load(cache_path, map_location="cpu")

            if isinstance(loaded, dict):
                self.samples = loaded["samples"]
            else:
                self.samples = loaded

            if max_samples is not None:
                self.samples = self.samples[:max_samples]

            print(f"[CACHE LOADED] {len(self.samples)} samples")

            return

        # ====================================================
        # BUILD CACHE
        # ====================================================

        print(f"[CACHE MISSING] Building {split} cache...")

        start_time = time.time()

        generator = HypothesisGenerator()

        encoder = HypothesisEncoder()

        ds = load_arc()

        if split not in ds:

            if split == "validation" and "test" in ds:
                split = "test"

            elif "validation" in ds:
                split = "validation"

            else:
                split = list(ds.keys())[0]

        raw = ds[split]

        count = 0

        for ex in tqdm(raw, desc=f"Building {split} cache"):

            try:

                question = ex["question"]

                # ============================================
                # ARC FORMAT
                # ============================================

                if isinstance(question, dict):

                    stem = question["stem"]

                    choices = question["choices"]

                    options = choices["text"]

                    labels = choices["label"]

                else:

                    stem = ex["question"]

                    choices = ex["choices"]

                    options = choices["text"]

                    labels = choices["label"]

                # ============================================
                # FILTERS
                # ============================================

                if len(options) < 2:
                    continue

                answer = ex["answerKey"]

                if answer not in labels:
                    continue

                y = labels.index(answer)

                # ============================================
                # HYPOTHESIS GENERATION
                # ============================================

                hypotheses = generator.generate(question=stem, options=options,)

                # ============================================
                # EMBEDDINGS
                # ============================================

                H = encoder.encode(hypotheses)

                O = encoder.encode(options)

                # ============================================
                # SAMPLE
                # ============================================

                sample = {

                    "question": stem,

                    "options": options,

                    "hypotheses": hypotheses,

                    "option_index": list(range(len(options))),

                    "H": H.cpu(),

                    "O": O.cpu(),

                    "y": y,
                }

                self.samples.append(sample)

                count += 1

                # ============================================
                # PERIODIC SAVE
                # ============================================

                if count % 50 == 0:

                    cache = {

                        "samples": self.samples,

                        "encoder": "MiniLM-L6-v2",

                        "generator": "Qwen2.5",

                        "version": 32,
                    }

                    torch.save(cache, cache_path)                

                    print(f"[AUTOSAVE] " f"{count} samples")

                if max_samples is not None and count >= max_samples:
                    break

            except Exception as e:

                print(f"[SKIP] {e}")

                continue

        # ====================================================
        # FINAL SAVE
        # ====================================================

        torch.save(self.samples, cache_path)

        elapsed = (time.time() - start_time) / 60

        print(f"[CACHE SAVED] " f"{cache_path}")

        print(f"[TOTAL SAMPLES] " f"{len(self.samples)}")

        print(f"[BUILD TIME] " f"{elapsed:.2f} min")

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        return self.samples[idx]


def collate_fn(batch, shuffle_options=True):
    """
    shuffle_options: randomly permutes each sample's option (and matching
    hypothesis) order so the model can't learn positional shortcuts from
    the small dataset. Pass shuffle_options=False for validation/eval so
    metrics are stable and reproducible.
    """

    max_h = max(x["H"].shape[0] for x in batch)

    max_o = max(x["O"].shape[0] for x in batch)

    dim = batch[0]["H"].shape[-1]

    Hs = []
    Os = []
    ys = []
    H_masks = []
    O_masks = []

    for sample in batch:

        H = sample["H"]
        O = sample["O"]
        y = sample["y"]

        if shuffle_options:

            n = O.shape[0]

            perm = torch.randperm(n)

            O = O[perm]

            # H and O share index order (one hypothesis per option),
            # so permute H the same way if they're aligned 1:1.
            if H.shape[0] == n:
                H = H[perm]

            y = int((perm == y).nonzero(as_tuple=True)[0].item())

        h_len = H.shape[0]
        o_len = O.shape[0]

        H_mask = H.new_zeros(max_h, dtype=torch.bool)

        O_mask = O.new_zeros(max_o, dtype=torch.bool)

        H_mask[:h_len] = True
        O_mask[:o_len] = True

        # ----------------------------------
        # PAD HYPOTHESES
        # ----------------------------------

        if H.shape[0] < max_h:

            pad = H.new_zeros(
                max_h - h_len,
                dim,
            )

            H = torch.cat([H, pad], dim=0)

        # ----------------------------------
        # PAD OPTIONS
        # ----------------------------------

        if O.shape[0] < max_o:

            pad = O.new_zeros(
                max_o - o_len,
                dim,
            )

            O = torch.cat([O, pad], dim=0)

        Hs.append(H)
        Os.append(O)

        ys.append(y)
        H_masks.append(H_mask)
        O_masks.append(O_mask)

    return {
        "H": torch.stack(Hs),
        "O": torch.stack(Os),
        "H_mask": torch.stack(H_masks),
        "O_mask": torch.stack(O_masks),
        "y": torch.tensor(
            ys,
            dtype=torch.long,
        ),
    }
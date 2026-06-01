import os
import torch

from torch.utils.data import Dataset

from benchmarks.arc import load_arc

from models.generator import HypothesisGenerator
from models.encoder import HypothesisEncoder


DIM = 384


class QAIRDataset(Dataset):

    def __init__(
        self,
        split="train",
        max_samples=500,
        cache_dir="./cache"
    ):

        self.samples = []

        os.makedirs(
            cache_dir,
            exist_ok=True
        )

        cache_path = os.path.join(
            cache_dir,
            f"arc_{split}.pt"
        )

        # ====================================================
        # LOAD CACHE
        # ====================================================

        if os.path.exists(cache_path):

            print(
                f"[CACHE FOUND] {cache_path}"
            )

            self.samples = torch.load(
                cache_path,
                map_location="cpu"
            )

            self.samples = self.samples[
                :max_samples
            ]

            print(
                f"[CACHE LOADED] "
                f"{len(self.samples)} samples"
            )

            return

        # ====================================================
        # BUILD CACHE
        # ====================================================

        print(
            f"[CACHE MISSING] Building {split} cache..."
        )

        generator = HypothesisGenerator()

        encoder = HypothesisEncoder()

        ds = load_arc()

        if split not in ds:

            if (
                split == "validation"
                and "test" in ds
            ):
                split = "test"

        raw = ds[split]

        count = 0

        for ex in raw:

            try:

                question = ex["question"]

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

                answer = ex["answerKey"]

                if answer not in labels:
                    continue

                y = labels.index(answer)

                # ============================================
                # HYPOTHESES
                # ============================================

                hypotheses = generator.generate(
                    stem,
                    options
                )

                # ============================================
                # EMBEDDINGS
                # ============================================

                H = encoder.encode(
                    hypotheses
                ).cpu()

                O = encoder.encode(
                    options
                ).cpu()

                sample = {

                    "question": stem,

                    "options": options,

                    "hypotheses": hypotheses,

                    "H": H,

                    "O": O,

                    "y": y

                }

                self.samples.append(sample)

                count += 1

                if count % 25 == 0:

                    print(
                        f"[{split}] "
                        f"{count}/{max_samples}"
                    )

                if count >= max_samples:
                    break

            except Exception as e:

                print(
                    f"[SKIP] {e}"
                )

        # ====================================================
        # SAVE CACHE
        # ====================================================

        torch.save(
            self.samples,
            cache_path
        )

        print(
            f"[CACHE SAVED] "
            f"{cache_path}"
        )

        print(
            f"[TOTAL SAMPLES] "
            f"{len(self.samples)}"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        return self.samples[idx]


# ============================================================
# COLLATE
# ============================================================

def collate_fn(batch):

    H = torch.stack(
        [x["H"] for x in batch]
    )

    O = torch.stack(
        [x["O"] for x in batch]
    )

    y = torch.tensor(
        [x["y"] for x in batch],
        dtype=torch.long
    )

    return {

        "H": H,

        "O": O,

        "y": y

    }
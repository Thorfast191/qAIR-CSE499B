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

        start_time = time.time()

        generator = HypothesisGenerator()

        encoder = HypothesisEncoder()

        ds = load_arc()

        if split not in ds:

            if (
                split == "validation"
                and "test" in ds
            ):
                split = "test"

            elif "validation" in ds:
                split = "validation"

            else:
                split = list(ds.keys())[0]

        raw = ds[split]

        count = 0

        for ex in tqdm(
            raw,
            desc=f"Building {split} cache"
        ):

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

                # ============================================
                # SAMPLE
                # ============================================

                sample = {

                    "question": stem,

                    "options": options,

                    "hypotheses": hypotheses,

                    "H": H,

                    "O": O,

                    "y": y

                }

                self.samples.append(
                    sample
                )

                count += 1

                # ============================================
                # PERIODIC SAVE
                # ============================================

                if count % 50 == 0:

                    torch.save(
                        self.samples,
                        cache_path
                    )

                    print(
                        f"[AUTOSAVE] "
                        f"{count} samples"
                    )

                if count >= max_samples:
                    break

            except Exception as e:

                print(
                    f"[SKIP] {e}"
                )

                continue

        # ====================================================
        # FINAL SAVE
        # ====================================================

        torch.save(
            self.samples,
            cache_path
        )

        elapsed = (
            time.time()
            - start_time
        ) / 60

        print(
            f"[CACHE SAVED] "
            f"{cache_path}"
        )

        print(
            f"[TOTAL SAMPLES] "
            f"{len(self.samples)}"
        )

        print(
            f"[BUILD TIME] "
            f"{elapsed:.2f} min"
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
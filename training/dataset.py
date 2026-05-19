import os
import random
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoModel
)

from datasets import load_dataset

# ============================================================
# DEVICE
# ============================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# EMBEDDING MODEL
# ============================================================

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(EMB_MODEL)

encoder = AutoModel.from_pretrained(
    EMB_MODEL
).to(device)

encoder.eval()

for p in encoder.parameters():
    p.requires_grad = False

DIM = encoder.config.hidden_size

# ============================================================
# EMBEDDING
# ============================================================

@torch.no_grad()
def embed_texts(texts):

    inp = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    ).to(device)

    out = encoder(**inp)

    hidden = out.last_hidden_state

    mask = inp["attention_mask"].unsqueeze(-1)

    pooled = (hidden * mask).sum(dim=1)

    counts = mask.sum(dim=1)

    pooled = pooled / counts.clamp(min=1e-9)

    pooled = F.normalize(pooled, dim=-1)

    return pooled.cpu()

# ============================================================
# HYPOTHESIS GENERATOR
# ============================================================

def generate_hypotheses(question, options):

    templates = [

        "The answer '{}' is supported by the evidence in the question.",

        "A contradiction would occur if '{}' were false under the described conditions.",

        "The reasoning process logically favors '{}' over the alternatives.",

        "Based on elimination and consistency, '{}' becomes the most plausible answer."

    ]

    hyps = []

    for i, opt in enumerate(options):

        template = templates[i % len(templates)]

        hyps.append(template.format(opt))

    return hyps

# ============================================================
# ARC
# ============================================================

def parse_arc(example):

    q = example["question"]

    if isinstance(q, dict):

        stem = q["stem"]

        choices = q["choices"]

        options = choices["text"]

        labels = choices["label"]

    else:

        stem = example["question"]

        choices = example["choices"]

        options = choices["text"]

        labels = choices["label"]

    answer = example["answerKey"]

    if answer not in labels:
        return None

    y = labels.index(answer)

    return {

        "question": stem,
        "options": options,
        "label": y

    }

# ============================================================
# COMMONSENSEQA
# ============================================================

def parse_commonsenseqa(example):

    question = example["question"]

    choices = example["choices"]

    options = choices["text"]

    labels = choices["label"]

    answer = example["answerKey"]

    if answer not in labels:
        return None

    y = labels.index(answer)

    return {

        "question": question,
        "options": options,
        "label": y

    }

# ============================================================
# STRATEGYQA
# ============================================================

def parse_strategyqa(example):

    question = example["question"]

    answer = example["answer"]

    options = ["No", "Yes"]

    y = 1 if answer else 0

    return {

        "question": question,
        "options": options,
        "label": y

    }

# ============================================================
# OPENBOOKQA
# ============================================================

def parse_openbookqa(example):

    question = example["question_stem"]

    choices = example["choices"]

    options = choices["text"]

    labels = choices["label"]

    answer = example["answerKey"]

    if answer not in labels:
        return None

    y = labels.index(answer)

    return {

        "question": question,
        "options": options,
        "label": y

    }

# ============================================================
# DATASET REGISTRY
# ============================================================

DATASET_REGISTRY = {

    "arc": {
        "loader": lambda: load_dataset(
            "ai2_arc",
            "ARC-Challenge"
        ),
        "parser": parse_arc
    },

    "commonsenseqa": {
        "loader": lambda: load_dataset(
            "commonsense_qa"
        ),
        "parser": parse_commonsenseqa
    },

    "strategyqa": {
        "loader": lambda: load_dataset(
            "strategyqa"
        ),
        "parser": parse_strategyqa
    },

    "openbookqa": {
        "loader": lambda: load_dataset(
            "openbookqa",
            "main"
        ),
        "parser": parse_openbookqa
    }
}

# ============================================================
# MAIN DATASET
# ============================================================

class QAIRDataset(Dataset):

    def __init__(
        self,
        benchmark="arc",
        split="train",
        max_samples=500,
        cache_dir="./data"
    ):

        self.samples = []

        self.benchmark = benchmark

        os.makedirs(cache_dir, exist_ok=True)

        cache_path = os.path.join(

            cache_dir,

            f"{benchmark}_{split}_{max_samples}.pt"

        )

        # ====================================================
        # CACHE
        # ====================================================

        if os.path.exists(cache_path):

            print(f"[Dataset] Loading cache: {cache_path}")

            self.samples = torch.load(cache_path)

            return

        # ====================================================
        # LOAD DATASET
        # ====================================================

        print(f"[Dataset] Loading benchmark: {benchmark}")

        registry = DATASET_REGISTRY[benchmark]

        dataset = registry["loader"]()

        parser = registry["parser"]

        # ====================================================
        # SPLIT FIX
        # ====================================================

        if split not in dataset:

            if split == "validation" and "test" in dataset:
                split = "test"

            elif split == "validation" and "validation" in dataset:
                split = "validation"

            else:
                split = list(dataset.keys())[0]

        raw = dataset[split]

        print(f"[Dataset] Processing {split}...")

        # ====================================================
        # PROCESS
        # ====================================================

        count = 0

        for ex in raw:

            try:

                parsed = parser(ex)

                if parsed is None:
                    continue

                q = parsed["question"]

                opts = parsed["options"]

                y = parsed["label"]

                # ============================================
                # FILTER
                # ============================================

                if len(opts) < 2:
                    continue

                if len(q.strip()) < 10:
                    continue

                # ============================================
                # HYPOTHESES
                # ============================================

                hyps = generate_hypotheses(q, opts)

                # ============================================
                # EMBEDDINGS
                # ============================================

                H = embed_texts(hyps)

                O = embed_texts(opts)

                sample = {

                    "H": H,
                    "O": O,
                    "y": y,
                    "question": q,
                    "options": opts,
                    "benchmark": benchmark

                }

                self.samples.append(sample)

                count += 1

                if count % 50 == 0:

                    print(
                        f"[{benchmark}] "
                        f"Processed {count}/{max_samples}"
                    )

                if count >= max_samples:
                    break

            except Exception as e:

                print(f"[SKIP] {e}")

                continue

        # ====================================================
        # SAVE CACHE
        # ====================================================

        torch.save(self.samples, cache_path)

        print(f"[Dataset] Saved cache: {cache_path}")

        print(
            f"[Dataset] Final: "
            f"{len(self.samples)} samples"
        )

    def __len__(self):

        return len(self.samples)

    def __getitem__(self, idx):

        return self.samples[idx]

# ============================================================
# COLLATE
# ============================================================

def collate_fn(batch):

    H = torch.stack([x["H"] for x in batch])

    O = torch.stack([x["O"] for x in batch])

    y = torch.tensor(
        [x["y"] for x in batch],
        dtype=torch.long
    )

    return {

        "H": H,
        "O": O,
        "y": y

    }

# ============================================================
# QUICK TEST
# ============================================================

if __name__ == "__main__":

    ds = QAIRDataset(

        benchmark="arc",

        split="train",

        max_samples=10

    )

    print(ds[0]["question"])

    print(ds[0]["options"])

    print(ds[0]["H"].shape)

    print(ds[0]["O"].shape)
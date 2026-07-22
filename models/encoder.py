import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel

from config import EMBEDDING_MODEL, EMBEDDING_DIM


class HypothesisEncoder:

    def __init__(
        self,
        model_name=EMBEDDING_MODEL,
        device="cuda"
    ):

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModel.from_pretrained(model_name).to(device)

        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.dim = self.model.config.hidden_size

        if model_name == EMBEDDING_MODEL:
            # Everything downstream (dataset.py, sample_inference.py, ...)
            # reads dim from config.EMBEDDING_DIM rather than this instance,
            # so a silent mismatch here would surface as a shape error far
            # away from its cause. Catch it at the source instead.
            assert self.dim == EMBEDDING_DIM, (
                f"{model_name} produces {self.dim}-dim embeddings but "
                f"config.EMBEDDING_DIM is {EMBEDDING_DIM}. Update config.py."
            )

        print(f"[ENCODER] Using {model_name}")
        print(f"[ENCODER] Embedding dimension: {self.dim}")

    @torch.no_grad()
    def encode(self, texts):

        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(self.device)

        outputs = self.model(**inputs)

        hidden = outputs.last_hidden_state

        mask = inputs["attention_mask"].unsqueeze(-1)

        pooled = (hidden * mask).sum(dim=1)

        counts = mask.sum(dim=1)

        pooled = pooled / counts.clamp(min=1e-9)

        pooled = F.normalize(pooled, dim=-1)

        return pooled

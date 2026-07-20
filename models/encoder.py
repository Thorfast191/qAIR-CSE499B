"""
FIXED ENCODER CONFIGURATION
Changes:
1. Changed embedding model from "all-MiniLM-L6-v2" (384-dim) to "all-mpnet-base-v2" (768-dim)
2. Larger embeddings provide more capacity for complex reasoning
3. MPNet is better for semantic understanding than MiniLM
4. This pairs with DIM=768 in dataset.py
"""

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


class HypothesisEncoder:

    def __init__(
        self, 
        # FIXED: Changed from "sentence-transformers/all-MiniLM-L6-v2" (384-dim)
        # to "sentence-transformers/all-mpnet-base-v2" (768-dim)
        # MPNet provides better semantic representation and 2x larger embedding space
        model_name="sentence-transformers/all-mpnet-base-v2",
        device="cuda"
    ):

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModel.from_pretrained(model_name).to(device)

        self.model.eval()

        for p in self.model.parameters():
            p.requires_grad = False

        self.dim = self.model.config.hidden_size
        
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

import torch
import torch.nn as nn

from models.validator import HypothesisValidator
from models.quantum_layer import QuantumEvolutionLayer
from models.persistent_reasoner import PersistentReasoner
from models.collapse import CollapseController


class QAIRvNext(nn.Module):

    def __init__(
        self,
        dim,
        use_quantum=True,
        use_validator=True,
        persistent_steps=3
    ):
        super().__init__()

        self.use_quantum = use_quantum
        self.use_validator = use_validator

        self.reasoner = PersistentReasoner(
            dim,
            steps=persistent_steps
        )

        if use_quantum:
            self.quantum = QuantumEvolutionLayer(dim)

        if use_validator:
            self.validator = HypothesisValidator(dim)

        self.collapse = CollapseController(dim)

        self.option_proj = nn.Linear(dim, dim)

    def forward(self, H, O):

        H, trajectory, attn = self.reasoner(H)

        if self.use_quantum:
            H = H + self.quantum(H)

        validator_out = None

        if self.use_validator:
            validator_out = self.validator(H, O)
            H = H - validator_out["energy"].unsqueeze(-1)

        E_h, collapse_loss = self.collapse(H)

        Oproj = self.option_proj(O)

        scores = torch.einsum("bkd,bod->bko", H, Oproj)

        final_scores = scores.mean(dim=1)

        return {
            "scores": final_scores,
            "hypothesis_energy": E_h,
            "collapse_loss": collapse_loss,
            "trajectory": trajectory,
            "attention": attn,
            "validator": validator_out
        }
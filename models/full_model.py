import torch
import torch.nn as nn

from models.validator import HypothesisValidator
from models.quantum_layer import QuantumEvolutionLayer
from models.persistent_reasoner import PersistentReasoner
from models.collapse import CollapseController
from models.answer_selector import EnergyAnswerSelector


class QAIRvNext(nn.Module):

    def __init__(self, dim, use_quantum=True, use_validator=True, persistent_steps=3):

        super().__init__()

        self.use_quantum = use_quantum

        self.use_validator = use_validator

        self.reasoner = PersistentReasoner(dim, steps=persistent_steps)

        if use_quantum:

            self.quantum = QuantumEvolutionLayer(dim)

        if use_validator:

            self.validator = HypothesisValidator(dim)

        self.collapse = CollapseController()

        self.selector = EnergyAnswerSelector(
            dim,
        )

    def forward(self, H, O):

        validator_out = None

        if self.use_validator:
            validator_out = self.validator(H, O)

            quality = torch.tanh(validator_out["energy"])
            H = H + (0.1 * quality.unsqueeze(-1))
        H, trajectory, attn = self.reasoner(H)

        quantum_energy = None

        if self.use_quantum:
            q_state, quantum_energy = self.quantum(H)
            H = H + q_state

        if quantum_energy is not None:

            collapse_energy = quantum_energy
        else:
            collapse_energy = torch.zeros(H.shape[0], H.shape[1], device=H.device)

        collapse_out = self.collapse(collapse_energy)

        scores = self.selector(H, O)

        collapse_probs = collapse_out["probabilities"]

        final_energy = (scores * collapse_probs.unsqueeze(-1)).sum(dim=1)

        final_scores = -final_energy

        return {
            "scores": final_scores,
            "hypothesis_energy": collapse_out["energy"],
            "collapse_loss": collapse_out["collapse_loss"],
            "collapse_probs": collapse_out["probabilities"],
            "entropy": collapse_out["entropy"],
            "diversity": collapse_out["diversity"],
            "trajectory": trajectory,
            "attention": attn,
            "validator": validator_out,
        }

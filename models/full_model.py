import torch.nn as nn

from models.validator import HypothesisValidator
from models.quantum_layer import QuantumEvolutionLayer
from models.persistent_reasoner import PersistentReasoner
from models.collapse import CollapseController
from models.answer_selector import EnergyAnswerSelector
from models.energy_fusion import EnergyFusion


class QAIRvNext(nn.Module):

    def __init__(
        self,
        dim,
        use_quantum=True,
        use_validator=True,
        persistent_steps=3,
        n_qubits=12,
    ):

        super().__init__()

        self.use_quantum = use_quantum
        self.use_validator = use_validator
        self.persistent_steps = persistent_steps  # saved for inference-time sanity checks

        self.reasoner = PersistentReasoner(dim, steps=persistent_steps)

        if use_quantum:
            self.quantum = QuantumEvolutionLayer(dim, n_qubits=n_qubits)

        if use_validator:
            self.validator = HypothesisValidator(dim)

        self.selector = EnergyAnswerSelector(dim)

        self.fusion = EnergyFusion()

        self.collapse = CollapseController()

    def forward(self, H, O, y=None):

        H, trajectory, interaction = self.reasoner(H)

        quantum_energy = None

        if self.use_quantum:

            from torch.amp import autocast

            with autocast(device_type="cuda", enabled=False):
                q_state, quantum_energy = self.quantum(H.float())

            H = H + q_state

        validator_out = None
        validator_energy = None
        potential = None

        if self.use_validator:

            validator_out = self.validator(H, O, y)

            potential = validator_out["potential"]

            validator_energy = validator_out["validator_energy"]

        selector = self.selector(H, O)

        answer_energy = selector["energy"]

        selector_confidence = selector["confidence"]

        collapse_energy = self.fusion(
            answer_energy,
            quantum_energy=quantum_energy,
            validator_energy=validator_energy,
        )

        collapse_out = self.collapse(collapse_energy)

        collapse_probs = collapse_out["probabilities"]

        final_energy = (answer_energy * collapse_probs.unsqueeze(-1)).sum(dim=1)

        final_scores = -final_energy

        return {
            "scores": final_scores,
            "answer_energy": answer_energy,
            "collapse_energy": collapse_energy,
            "collapse_probs": collapse_probs,
            "selector_confidence": selector_confidence,
            "quantum_energy": quantum_energy,
            "collapse_loss": collapse_out["collapse_loss"],
            "entropy": collapse_out["entropy"],
            "diversity": collapse_out["diversity"],
            "spread": collapse_out["spread"],
            "peak": collapse_out["peak"],
            "validator_potential": potential,
            "trajectory": trajectory,
            "attention": interaction,
            "validator": validator_out,
        }
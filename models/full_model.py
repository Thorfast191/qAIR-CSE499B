import torch
import torch.nn as nn

from models.validator import HypothesisValidator
from models.quantum_layer import QuantumEvolutionLayer
from models.persistent_reasoner import PersistentReasoner
from models.collapse import CollapseController
from models.answer_selector import EnergyAnswerSelector


class QAIRvNext(nn.Module):

    def __init__(
        self,
        dim,
        use_quantum=True,
        use_validator=True,
        persistent_steps=3,
    ):

        super().__init__()

        self.use_quantum = use_quantum
        self.use_validator = use_validator

        self.reasoner = PersistentReasoner(
            dim,
            steps=persistent_steps,
        )

        if use_quantum:
            self.quantum = QuantumEvolutionLayer(dim)

        if use_validator:
            self.validator = HypothesisValidator(dim)

        self.selector = EnergyAnswerSelector(dim)

        self.energy_alpha = nn.Parameter(torch.tensor(0.0))

        self.collapse = CollapseController()

    def forward(self, H, O, y=None):

        ####################################################
        # Persistent Reasoning
        ####################################################

        H, trajectory, interaction = self.reasoner(H)

        ####################################################
        # Quantum Evolution
        ####################################################

        quantum_energy = None

        if self.use_quantum:

            q_state, quantum_energy = self.quantum(H)

            H = H + q_state

        ####################################################
        # Validator AFTER reasoning
        ####################################################

        validator_out = None
        validator_energy = None
        potential = None

        if self.use_validator:

            validator_out = self.validator(
                H,
                O,
                y,
            )

            potential = validator_out["potential"]

            validator_energy = validator_out[
                "validator_energy"
            ]

        ####################################################
        # Adaptive Answer Selector
        ####################################################

        selector = self.selector(
            H,
            O,
        )

        answer_energy = selector["energy"]

        selector_confidence = selector["confidence"]

        ####################################################
        # Free Energy over Answers
        ####################################################

        tau = 1.0

        selector_energy = -tau * torch.logsumexp(
            -answer_energy / tau,
            dim=-1,
        )

        ####################################################
        # Collapse Energy
        ####################################################

        collapse_energy = selector_energy

        if quantum_energy is not None:

            quantum_energy = (
                quantum_energy
                - quantum_energy.mean(dim=1, keepdim=True)
            )

            quantum_energy = (
                quantum_energy
                /
                (
                    quantum_energy.std(
                        dim=1,
                        keepdim=True,
                    )
                    + 1e-6
                )
            )

            alpha = torch.sigmoid(
                self.energy_alpha
            )

            collapse_energy = (
                alpha * collapse_energy
                +
                (1.0 - alpha) * quantum_energy
            )

        ####################################################
        # Validator Guidance
        ####################################################

        if validator_energy is not None:

            validator_energy = (
                validator_energy
                - validator_energy.mean(
                    dim=1,
                    keepdim=True,
                )
            )

            validator_energy = (
                validator_energy
                /
                (
                    validator_energy.std(
                        dim=1,
                        keepdim=True,
                    )
                    + 1e-6
                )
            )

            collapse_energy = (
                collapse_energy
                +
                0.30 * validator_energy
            )

        ####################################################
        # Collapse
        ####################################################

        collapse_out = self.collapse(
            collapse_energy
        )

        collapse_probs = collapse_out[
            "probabilities"
        ]

        ####################################################
        # Final Answer Energy
        ####################################################

        final_energy = (
            answer_energy
            * collapse_probs.unsqueeze(-1)
        ).sum(dim=1)

        final_scores = -final_energy

        ####################################################
        # Return
        ####################################################

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
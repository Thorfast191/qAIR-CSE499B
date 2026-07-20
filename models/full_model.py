"""
UPDATED FULL MODEL with Enhanced Quantum Layer
For Quantum-Inspired Reasoning Research

This version keeps quantum layer as core component but makes it
much more powerful and better integrated.
"""

import torch
import torch.nn as nn

from models.validator import HypothesisValidator
from models.quantum_layer_ENHANCED import QuantumEvolutionLayerEnhanced, ImprovedQuantumIntegration
from models.persistent_reasoner import PersistentReasoner
from models.collapse import CollapseController
from models.answer_selector import EnergyAnswerSelector


class QAIRvNextQuantumFocused(nn.Module):
    """
    qAIR-vNext optimized for Quantum Reasoning Research
    
    Key features:
    - Enhanced quantum layer (multi-scale, adaptive)
    - Intelligent quantum energy fusion
    - Better quantum-classical integration
    - Focused on quantum-inspired reasoning
    """

    def __init__(
        self,
        dim,
        use_quantum=True,
        use_validator=True,
        persistent_steps=5,
        quantum_enhanced=True,
    ):

        super().__init__()

        self.use_quantum = use_quantum
        self.use_validator = use_validator
        self.quantum_enhanced = quantum_enhanced

        self.reasoner = PersistentReasoner(
            dim,
            steps=persistent_steps,
        )

        if use_quantum:
            if quantum_enhanced:
                # Use enhanced quantum layer for research
                self.quantum = QuantumEvolutionLayerEnhanced(dim)
                self.quantum_integration = ImprovedQuantumIntegration(dim)
            else:
                # Fall back to original quantum layer
                from models.quantum_layer import QuantumEvolutionLayer
                self.quantum = QuantumEvolutionLayer(dim)
                self.quantum_integration = None

        if use_validator:
            self.validator = HypothesisValidator(dim)

        self.selector = EnergyAnswerSelector(dim)

        # Learn balance between quantum and classical reasoning
        self.energy_alpha = nn.Parameter(torch.tensor(0.0))

        self.collapse = CollapseController()

    def forward(self, H, O, y=None):

        ####################################################
        # Persistent Reasoning
        ####################################################

        H, trajectory, interaction = self.reasoner(H)

        ####################################################
        # Enhanced Quantum Evolution
        ####################################################

        quantum_energy = None

        if self.use_quantum:

            from torch.amp import autocast

            with autocast(device_type="cuda", enabled=False):
                q_state, quantum_energy = self.quantum(H.float())

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

        selector_energy = torch.min(
            answer_energy / tau,
            dim=-1,
        ).values

        ####################################################
        # Collapse Energy (with quantum integration)
        ####################################################

        collapse_energy = selector_energy

        # IMPROVED: Intelligent quantum energy fusion
        if quantum_energy is not None:
            if self.quantum_integration is not None:
                # Use enhanced fusion
                collapse_energy = self.quantum_integration.fuse_quantum_energy(
                    selector_energy,
                    quantum_energy,
                    H,
                )
            else:
                # Use original fusion
                alpha = torch.sigmoid(self.energy_alpha)
                collapse_energy = (
                    alpha * collapse_energy
                    + (1.0 - alpha) * quantum_energy
                )

        ####################################################
        # Validator Guidance
        ####################################################

        if validator_energy is not None:

            collapse_energy = (
                collapse_energy
                + 0.30 * validator_energy
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
        # Return (with quantum diagnostics)
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

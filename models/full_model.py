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

        self.collapse = CollapseController()

    def forward(self, H, O):

        validator_out = None
        potential = None

        # ---------------------------------------------------
        # Validator
        # ---------------------------------------------------

        if self.use_validator:

            validator_out = self.validator(H, O)

            potential = validator_out["potential"]

        # ---------------------------------------------------
        # Persistent reasoning
        # ---------------------------------------------------

        print("=" * 50)

        print("Input H:", H.shape)

        print("Input O:", O.shape)

        if potential is not None:

            print("Potential:", potential.shape)

        H, trajectory, attn = self.reasoner(
            H,
            potential,
        )

        # ---------------------------------------------------
        # Quantum evolution
        # ---------------------------------------------------

        quantum_energy = None

        if self.use_quantum:

            q_state, quantum_energy = self.quantum(H)

            H = H + q_state

        # ---------------------------------------------------
        # Answer Hamiltonian
        # ---------------------------------------------------

        answer_energy = self.selector(
            H,
            O,
        )                       # (B,K,N)

        # ---------------------------------------------------
        # Collapse Energy
        #
        # Lower energy = better hypothesis
        # ---------------------------------------------------

        selector_energy = answer_energy.mean(
            dim=-1
        )                       # (B,K)

        if quantum_energy is not None:

            collapse_energy = (
                selector_energy
                + quantum_energy
            )

        else:

            collapse_energy = selector_energy

        collapse_out = self.collapse(
            collapse_energy
        )

        collapse_probs = collapse_out[
            "probabilities"
        ]                       # (B,K)

        # ---------------------------------------------------
        # Final Answer Energy
        # ---------------------------------------------------

        final_energy = (
            answer_energy
            * collapse_probs.unsqueeze(-1)
        ).sum(
            dim=1
        )                       # (B,N)

        final_scores = -final_energy

        return {

            "scores": final_scores,

            "answer_energy": answer_energy,

            "collapse_energy": collapse_energy,

            "quantum_energy": quantum_energy,

            "collapse_probs": collapse_probs,

            "collapse_loss": collapse_out[
                "collapse_loss"
            ],

            "entropy": collapse_out[
                "entropy"
            ],

            "diversity": collapse_out[
                "diversity"
            ],

            "spread": collapse_out[
                "spread"
            ],

            "peak": collapse_out[
                "peak"
            ],

            "validator_potential": potential,

            "trajectory": trajectory,

            "attention": attn,

            "validator": validator_out,
        }
import torch
import torch.nn as nn

from models.validator import HypothesisValidator
from models.quantum_layer import QuantumEvolutionLayer
from models.classical_control_layer import ClassicalControlLayer
from models.persistent_reasoner import PersistentReasoner
from models.collapse import CollapseController
from models.answer_selector import EnergyAnswerSelector
from models.energy_fusion import EnergyFusion

BACKENDS = ("quantum", "classical_control", "none")


class QAIRvNext(nn.Module):

    def __init__(
        self,
        dim,
        use_quantum=True,
        use_validator=True,
        persistent_steps=3,
        n_qubits=12,
        backend=None,
        use_question=True,
        validator_feedback=True,
        keep_trajectory=False,
        verbose=False,
    ):

        super().__init__()

        # backend selects what occupies the quantum-layer slot:
        #   "quantum"           -- QuantumEvolutionLayer (PennyLane circuit)
        #   "classical_control" -- ClassicalControlLayer (parameter-matched
        #                          classical MLP in the same slot -- see
        #                          models/classical_control_layer.py)
        #   "none"              -- no bottleneck layer at all
        # use_quantum is kept as a backward-compatible alias (True ->
        # "quantum", False -> "none") -- only used when backend isn't
        # given explicitly, so existing call sites keep working. The
        # submodule is stored as self.quantum regardless of which
        # backend is chosen (not e.g. self.bottleneck) so that
        # training/train.py's quantum./validator.-prefixed lower-LR
        # parameter group applies identically to both backends -- the
        # comparison between them should differ only in what's inside
        # that slot, not in how it's optimized.
        if backend is None:
            backend = "quantum" if use_quantum else "none"

        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")

        self.backend = backend
        self.use_quantum = backend != "none"
        self.use_validator = use_validator
        self.use_question = use_question
        self.persistent_steps = persistent_steps  # saved for inference-time sanity checks

        # The validator computes a `potential` field that was never
        # actually fed back into the reasoner -- QAIRvNext.forward called
        # self.reasoner(H) with no potential argument, even though
        # PersistentReasoner.forward(H, potential=None) accepted one. The
        # validator therefore only reached the answer through
        # validator_energy in EnergyFusion, never through the reasoning
        # trajectory. With feedback on, the reasoner runs twice: once to
        # get hypotheses good enough to validate, then again guided by
        # the resulting potential.
        self.validator_feedback = validator_feedback and use_validator

        self.reasoner = PersistentReasoner(
            dim, steps=persistent_steps, keep_trajectory=keep_trajectory
        )

        if backend == "quantum":
            self.quantum = QuantumEvolutionLayer(dim, n_qubits=n_qubits, verbose=verbose)
        elif backend == "classical_control":
            self.quantum = ClassicalControlLayer(
                dim, n_qubits=n_qubits, verbose=verbose
            )

        if use_validator:
            self.validator = HypothesisValidator(dim, use_question=use_question)

        self.selector = EnergyAnswerSelector(dim, use_question=use_question)

        self.fusion = EnergyFusion()

        self.collapse = CollapseController()

    def _run_quantum(self, H):

        if H.is_cuda:

            # Only construct a "cuda" autocast context on an actual CUDA
            # tensor -- on CPU/MPS this context wouldn't do anything
            # anyway (autocast is disabled either way in the training
            # loop for non-CUDA devices), and some PyTorch versions
            # validate device_type at construction even when disabled.
            from torch.amp import autocast

            with autocast(device_type="cuda", enabled=False):
                return self.quantum(H.float())

        return self.quantum(H.float())

    def forward(self, H, O, Q=None, y=None, H_mask=None, O_mask=None):
        """
        H      : (B, K, D) hypothesis embeddings
        O      : (B, N, D) option embeddings
        Q      : (B, D)    question embedding (required when use_question)
        y      : (B,)      labels, training only -- used solely to build the
                           validator's BCE target, never to compute energy
        H_mask : (B, K) bool, True for real hypotheses
        O_mask : (B, N) bool, True for real options

        Masks matter: collate_fn pads ragged option/hypothesis counts with
        zeros, and before the audit those masks were computed and then
        never passed to the model, so zero-padded options participated in
        every reduction and could be returned as the argmax prediction.
        """

        H, trajectory, interaction = self.reasoner(H, mask=H_mask)

        quantum_energy = None

        if self.use_quantum:

            q_state, quantum_energy = self._run_quantum(H)

            H = H + q_state

        validator_out = None
        validator_energy = None
        potential = None

        if self.use_validator:

            validator_out = self.validator(
                H, O, Q=Q, y=y, H_mask=H_mask, O_mask=O_mask
            )

            potential = validator_out["potential"]

            validator_energy = validator_out["validator_energy"]

            if self.validator_feedback:

                # Second reasoning pass, now guided by the validator's
                # potential -- this is the feedback edge the architecture
                # was described as having but did not.
                H, trajectory2, interaction = self.reasoner(
                    H, potential=potential, mask=H_mask
                )

                trajectory = trajectory + trajectory2

                if self.use_quantum:
                    q_state, quantum_energy = self._run_quantum(H)
                    H = H + q_state

                validator_out = self.validator(
                    H, O, Q=Q, y=y, H_mask=H_mask, O_mask=O_mask
                )

                potential = validator_out["potential"]

                validator_energy = validator_out["validator_energy"]

        selector = self.selector(H, O, Q=Q)

        answer_energy = selector["energy"]

        selector_confidence = selector["confidence"]

        collapse_energy = self.fusion(
            answer_energy,
            quantum_energy=quantum_energy,
            validator_energy=validator_energy,
            option_mask=O_mask,
        )

        collapse_out = self.collapse(collapse_energy, mask=H_mask)

        collapse_probs = collapse_out["probabilities"]

        # Probability-weighted marginalization over hypotheses: the
        # superposition survives to the final score rather than being
        # argmax'd away.
        final_energy = (answer_energy * collapse_probs.unsqueeze(-1)).sum(dim=1)

        final_scores = -final_energy

        if O_mask is not None:
            # Padded options must never be predicted or contribute
            # probability mass to the cross-entropy.
            neg_inf = torch.finfo(final_scores.dtype).min / 2
            final_scores = final_scores.masked_fill(~O_mask, neg_inf)

        return {
            "scores": final_scores,
            "answer_energy": answer_energy,
            "collapse_energy": collapse_energy,
            "collapse_probs": collapse_probs,
            "selector_confidence": selector_confidence,
            "quantum_energy": quantum_energy,
            "entropy": collapse_out["entropy"],
            "diversity": collapse_out["diversity"],
            "spread": collapse_out["spread"],
            "peak": collapse_out["peak"],
            "validator_potential": potential,
            "trajectory": trajectory,
            "attention": interaction,
            "validator": validator_out,
            "H_mask": H_mask,
            "O_mask": O_mask,
            # Post-reasoning hypotheses, so evaluate.py can distinguish
            # "the hypotheses merged into one vector" from "the selector
            # learned to ignore them". Both make the multi-hypothesis
            # mechanism inert, but they have different fixes.
            "H_reasoned": H,
        }

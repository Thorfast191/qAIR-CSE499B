import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import (
    EMBEDDING_DIM,
    MODEL_DIM,
    PHASE_MODE,
    PHASE_SOURCE,
    MIXING,
    USE_LLM_PRIOR,
    QUANTUM_DEVICE,
)

from models.validator import HypothesisValidator
from models.quantum_layer import QuantumEvolutionLayer
from models.classical_control_layer import ClassicalControlLayer
from models.persistent_reasoner import PersistentReasoner
from models.collapse import CollapseController
from models.answer_selector import EnergyAnswerSelector, bounded_phase
from models.energy_fusion import EnergyFusion
from models.interference import CoherentCollapse

BACKENDS = ("quantum", "classical_control", "none")


def _masked_zscore(x, mask, eps=1e-5):
    """Per-sample standardization over the option axis, ignoring padding."""

    m = mask.to(x.dtype) if mask is not None else torch.ones_like(x)

    n = m.sum(dim=1, keepdim=True).clamp(min=1.0)

    mean = (x * m).sum(dim=1, keepdim=True) / n

    var = (((x - mean) ** 2) * m).sum(dim=1, keepdim=True) / n

    return ((x - mean) / torch.sqrt(var + eps)) * m


class QAIRvNext(nn.Module):
    """
    qAIR-v45.

        Q, H, O  ->  in_proj (384 -> MODEL_DIM)
                 ->  PersistentReasoner        (orthogonal mixing)
                 ->  Quantum / classical layer (state + energy + PHASE)
                 ->  HypothesisValidator       (-> potential, fed back)
                 ->  EnergyAnswerSelector      (E_kn + compatibility phase)
                 ->  EnergyFusion              (-> E_k)
                 ->  CollapseController        (per-sample temperature)
                 ->  CoherentCollapse          (complex amplitudes,
                                                |sum_k c_k v_kn|^2)
                 ->  + LLM log-prior           (optional, ablatable)

    Three things are different from v44 in ways that matter, all of them
    responses to specific measured failures rather than tuning:

    1. **Width.** v44 ran 10.58M parameters over 800 training examples --
       13,223 per example -- and its best validation accuracy was at
       epoch 1, declining monotonically after while train loss fell 54%.
       Everything past `in_proj` now runs at MODEL_DIM (128), roughly 1.1M
       parameters. Below that ratio, "hypotheses don't help" and "the
       model memorizes before it learns to use them" are not
       distinguishable, so no mechanistic claim from v44 was safe.

    2. **Evidence channels.** The cache now carries a supporting AND an
       attacking hypothesis per option, each tagged with its polarity and
       the option it argues about (`align`). v44's support-only
       hypotheses were measured at argmax diag(H.O) = 0.2481 against a
       chance of 0.2500 -- literally zero discriminative signal, because
       asking a competent LLM to justify all four options gets all four
       justified.

    3. **Genuine interference.** The final distribution is
       |sum_k c_k v_kn|^2 over complex amplitudes, so hypotheses can
       cancel. See models/interference.py. `phase_mode="classical"`
       recovers v44's real-amplitude mixture as a control, and
       `interference_ratio` is reported every epoch so an inert phase
       channel shows up as a number rather than hiding behind a plausible
       accuracy.

    The LLM log-prior (`use_llm_prior`) deserves its own warning. It adds
    the generator's own cached per-option log-likelihood to the final
    score. That is legitimate -- it is the only channel by which the LLM's
    judgement reaches the answer without being squeezed through a frozen
    384-d sentence encoder -- but it also means the model can score well
    while ignoring every hypothesis. Always read
    `evaluation/input_ablation.py` alongside accuracy, and always run the
    `use_llm_prior=False` arm before attributing a gain to reasoning.
    """

    def __init__(
        self,
        dim=EMBEDDING_DIM,
        model_dim=MODEL_DIM,
        use_quantum=True,
        use_validator=True,
        persistent_steps=3,
        n_qubits=12,
        backend=None,
        use_question=True,
        validator_feedback=True,
        keep_trajectory=False,
        phase_mode=PHASE_MODE,
        phase_source=PHASE_SOURCE,
        mixing=MIXING,
        use_llm_prior=USE_LLM_PRIOR,
        use_attack=True,
        quantum_device=QUANTUM_DEVICE,
        verbose=False,
    ):

        super().__init__()

        # backend selects what occupies the quantum-layer slot:
        #   "quantum"           -- QuantumEvolutionLayer (PennyLane circuit)
        #   "classical_control" -- ClassicalControlLayer (parameter-matched
        #                          classical MLP in the same slot)
        #   "none"              -- no bottleneck layer at all
        # use_quantum is a backward-compatible alias (True -> "quantum",
        # False -> "none") used only when backend isn't given explicitly.
        # The submodule is stored as self.quantum regardless of backend so
        # training/train.py's quantum./validator.-prefixed lower-LR
        # parameter group applies identically to both -- the comparison
        # between them should differ only in what's inside that slot, not
        # in how it is optimized.
        if backend is None:
            backend = "quantum" if use_quantum else "none"

        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")

        d = model_dim or dim

        self.embed_dim = dim
        self.model_dim = d

        self.backend = backend
        self.use_quantum = backend != "none"
        self.use_validator = use_validator
        self.use_question = use_question
        self.use_llm_prior = use_llm_prior
        self.use_attack = use_attack
        self.phase_mode = phase_mode
        self.phase_source = phase_source
        self.mixing = mixing
        self.quantum_device = quantum_device
        self.persistent_steps = persistent_steps

        # The validator computes a `potential` field that was never fed
        # back into the reasoner before v44: forward() called
        # self.reasoner(H) with no potential argument even though
        # PersistentReasoner accepted one, so the validator only ever
        # reached the answer through validator_energy. With feedback on,
        # the reasoner runs twice: once to get hypotheses good enough to
        # validate, then again guided by the resulting potential.
        self.validator_feedback = validator_feedback and use_validator

        # ------------------------------------------------------------
        # Input projection: 384-d frozen embeddings -> model width
        # ------------------------------------------------------------

        if d == dim:
            self.in_proj = nn.Identity()
        else:
            self.in_proj = nn.Sequential(nn.Linear(dim, d), nn.LayerNorm(d))

        # Marks a hypothesis as supporting (+1) or attacking (-1) inside
        # the vector itself, so the reasoner's mixing can tell the two
        # channels apart rather than treating them as interchangeable.
        self.polarity_embed = nn.Parameter(torch.zeros(d))

        self.reasoner = PersistentReasoner(
            d, steps=persistent_steps, keep_trajectory=keep_trajectory,
            mixing=mixing,
        )

        if backend == "quantum":
            self.quantum = QuantumEvolutionLayer(
                d, n_qubits=n_qubits, verbose=verbose,
                quantum_device=quantum_device,
            )
        elif backend == "classical_control":
            self.quantum = ClassicalControlLayer(d, n_qubits=n_qubits, verbose=verbose)

        if use_validator:
            self.validator = HypothesisValidator(d, use_question=use_question)

        self.selector = EnergyAnswerSelector(d, use_question=use_question)

        self.fusion = EnergyFusion()

        self.collapse = CollapseController()

        self.interference = CoherentCollapse(phase_mode=phase_mode)

        # Classical fallback phase head, used when phase_source ==
        # "learned" or when there is no bottleneck layer to read a phase
        # off. One linear map to one bounded scalar per hypothesis.
        self.phase_head = nn.Linear(d, 1)

        # Relative phase between the two evidence directions, initialized
        # to pi.
        #
        # This is the one piece of physics the architecture actually
        # asserts, and it is worth stating plainly: an objection to
        # option n is the NEGATION of the support for option n, so the
        # natural relative phase between them is pi, where their
        # amplitudes cancel. Without it the phase channel starts
        # essentially uniform -- measured at initialization, phases from
        # both the circuit and the learned head are concentrated tightly
        # enough that the coherent sum is purely constructive and
        # `destructive_fraction` is exactly 0. The mechanism the whole
        # v45 claim rests on would begin dead and have to discover
        # cancellation from a gradient signal that only exists once
        # cancellation happens.
        #
        # It is a learnable parameter, not a constant: if support and
        # attack turn out not to be opposites in any useful sense, the
        # model can drive it to 0 and recover the in-phase behaviour.
        # `phase_mode="zero"` and `phase_mode="classical"` bypass it
        # entirely, so it is covered by the existing controls.
        self.polarity_phase = nn.Parameter(torch.tensor(math.pi))

        # Weight on the cached LLM log-prior. softplus keeps it
        # non-negative (a prior the model can trust or ignore, but never
        # invert); softplus(-1.0) = 0.313 at init -- present, not
        # dominant.
        self.llm_prior_weight = nn.Parameter(torch.tensor(-1.0))

    def _run_quantum(self, H):

        if H.is_cuda:

            # Only construct a "cuda" autocast context on an actual CUDA
            # tensor -- on CPU/MPS this context wouldn't do anything
            # anyway (autocast is disabled either way in the training
            # loop for non-CUDA devices), and some PyTorch versions
            # validate device_type at construction even when disabled.
            #
            # Forcing fp32 here applies regardless of which PennyLane
            # device is attached (see config.QUANTUM_DEVICE) -- PennyLane's
            # autograd through lightning.qubit isn't mixed-precision safe,
            # and lightning.gpu's cuStateVec backend is fp32/fp64
            # statevectors, not fp16/bf16, so autocast has nothing valid to
            # do to either one.
            from torch.amp import autocast

            with autocast(device_type="cuda", enabled=False):
                return self.quantum(H.float())

        return self.quantum(H.float())

    def forward(self, H, O, Q=None, y=None, H_mask=None, O_mask=None,
                polarity=None, align=None, llm_logprob=None):
        """
        H           : (B, K, D_embed) hypothesis embeddings, K = 2N
        O           : (B, N, D_embed) option embeddings
        Q           : (B, D_embed)    question embedding
        y           : (B,)  labels, training only -- used solely to build
                      the validator's BCE target, never to compute energy
        H_mask      : (B, K) bool, True for real hypotheses
        O_mask      : (B, N) bool, True for real options
        polarity    : (B, K) +1 support / -1 attack
        align       : (B, K, N) 1 where hypothesis k is about option n
        llm_logprob : (B, N) cached per-option LLM log-likelihood

        Masks matter: collate_fn pads ragged option/hypothesis counts with
        zeros, and before the v44 audit those masks were computed and then
        never passed to the model, so zero-padded options participated in
        every reduction and could be returned as the argmax prediction.
        """

        B, K, _ = H.shape
        N = O.shape[1]

        if H_mask is None:
            H_mask = torch.ones(B, K, dtype=torch.bool, device=H.device)

        if O_mask is None:
            O_mask = torch.ones(B, N, dtype=torch.bool, device=O.device)

        if polarity is None:
            polarity = torch.ones(B, K, device=H.device, dtype=H.dtype)

        # Support-only ablation: drop the attacking channel without
        # touching the cache, so the comparison is over identical data.
        if not self.use_attack:
            H_mask = H_mask & (polarity >= 0)

        # ------------------------------------------------------------
        # Project into model width
        # ------------------------------------------------------------

        H = self.in_proj(H)
        O = self.in_proj(O)
        Q = self.in_proj(Q) if Q is not None else None

        H = H + polarity.unsqueeze(-1) * self.polarity_embed

        H = H * H_mask.unsqueeze(-1).to(H.dtype)

        # ------------------------------------------------------------
        # Reason
        # ------------------------------------------------------------

        H, trajectory, interaction = self.reasoner(H, mask=H_mask)

        quantum_energy = None
        quantum_phase = None

        if self.use_quantum:

            q_state, quantum_energy, quantum_phase = self._run_quantum(H)

            H = H + q_state

        validator_out = None
        validator_energy = None
        potential = None

        if self.use_validator:

            validator_out = self.validator(
                H, O, Q=Q, y=y, H_mask=H_mask, O_mask=O_mask,
                align=align, polarity=polarity,
            )

            potential = validator_out["potential"]

            validator_energy = validator_out["validator_energy"]

            if self.validator_feedback:

                # Second reasoning pass, guided by the validator's
                # potential -- the feedback edge the architecture was
                # described as having but did not until v44.
                H, trajectory2, interaction = self.reasoner(
                    H, potential=potential, mask=H_mask
                )

                trajectory = trajectory + trajectory2

                if self.use_quantum:
                    q_state, quantum_energy, quantum_phase = self._run_quantum(H)
                    H = H + q_state

                validator_out = self.validator(
                    H, O, Q=Q, y=y, H_mask=H_mask, O_mask=O_mask,
                    align=align, polarity=polarity,
                )

                potential = validator_out["potential"]

                validator_energy = validator_out["validator_energy"]

        # ------------------------------------------------------------
        # Pairwise energy + compatibility phase
        # ------------------------------------------------------------

        selector = self.selector(
            H, O, Q=Q, align=align, polarity=polarity,
            H_mask=H_mask, O_mask=O_mask,
        )

        answer_energy = selector["energy"]

        collapse_energy = self.fusion(
            answer_energy,
            quantum_energy=quantum_energy,
            validator_energy=validator_energy,
            option_mask=O_mask,
        )

        collapse_out = self.collapse(collapse_energy, mask=H_mask)

        # ------------------------------------------------------------
        # Hypothesis phase
        # ------------------------------------------------------------

        if self.phase_source == "circuit" and quantum_phase is not None:
            phase_h = quantum_phase
        else:
            phase_h = bounded_phase(self.phase_head(H).squeeze(-1))

        # +0 for support (polarity +1), +polarity_phase for attack (-1).
        phase_h = phase_h + 0.5 * (1.0 - polarity) * self.polarity_phase

        phase_h = phase_h * H_mask.to(phase_h.dtype)

        # ------------------------------------------------------------
        # Coherent collapse -- |sum_k c_k v_kn|^2
        # ------------------------------------------------------------

        coherent = self.interference(
            collapse_energy,
            answer_energy,
            phase_h=phase_h,
            phase_kn=selector["phase"],
            H_mask=H_mask,
            O_mask=O_mask,
            hyp_temperature=collapse_out["temperature"],
        )

        scores = coherent["log_probs"]

        # ------------------------------------------------------------
        # LLM log-prior (explicit, weighted, ablatable)
        # ------------------------------------------------------------

        llm_term = None

        if self.use_llm_prior and llm_logprob is not None:

            # Standardized per sample: raw mean-token log-likelihoods sit
            # around -2 to -8 with a scale that varies with option length,
            # and an unnormalized additive term at that scale would swamp
            # log_probs outright.
            llm_term = _masked_zscore(llm_logprob, O_mask)

            scores = scores + F.softplus(self.llm_prior_weight) * llm_term

        # Padded options must never be predicted or carry probability
        # mass in the cross-entropy.
        neg_inf = torch.finfo(scores.dtype).min / 2
        scores = scores.masked_fill(~O_mask, neg_inf)

        # Renormalized, so `scores` is a genuine log-probability vector
        # and calibration metrics (evaluation ECE) are meaningful.
        #
        # The floor is not cosmetic: without it a padded option lands at
        # exactly -inf, and label smoothing -- which puts a little mass on
        # EVERY class -- multiplies that by epsilon and returns inf. The
        # loss then NaNs on any batch containing a 3-option question,
        # which ARC has. At -30 a padded option carries 9e-14 of the
        # probability mass, i.e. nothing, while staying finite.
        scores = torch.log_softmax(scores, dim=1).clamp(min=-30.0)

        collapse_probs = coherent["collapse_probs"]

        entropy = -(
            collapse_probs * torch.log(collapse_probs + 1e-8)
        ).sum(dim=1).mean()

        return {
            "scores": scores,
            "answer_energy": answer_energy,
            "collapse_energy": collapse_energy,
            "collapse_probs": collapse_probs,
            "quantum_energy": quantum_energy,
            "quantum_phase": quantum_phase,
            "phase_h": phase_h,
            "phase_kn": selector["phase"],
            "interference": coherent["interference"],
            "interference_ratio": coherent["interference_ratio"],
            "phase_effect": coherent["phase_effect"],
            "destructive_fraction": coherent["destructive_fraction"],
            "classical_weights": coherent["classical"],
            "coherent_weights": coherent["coherent"],
            "llm_prior": llm_term,
            "llm_prior_weight": F.softplus(self.llm_prior_weight).detach(),
            "entropy": entropy,
            "diversity": collapse_out["diversity"],
            "spread": collapse_out["spread"],
            "peak": collapse_probs.max(dim=1).values.mean(),
            "collapse_confidence": collapse_out["confidence"],
            "validator_potential": potential,
            "trajectory": trajectory,
            "attention": interaction,
            "validator": validator_out,
            "H_mask": H_mask,
            "O_mask": O_mask,
            # Post-reasoning hypotheses, so training/evaluate.py can
            # distinguish "the hypotheses merged into one vector" from
            # "the selector learned to ignore them". Both make the
            # multi-hypothesis mechanism inert, with different fixes.
            "H_reasoned": H,
        }

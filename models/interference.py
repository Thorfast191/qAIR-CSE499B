"""
Coherent collapse -- complex amplitudes and real interference.

Why this module exists
----------------------
Every "quantum" operation in v44 reduced to a classical one with a
physics name attached. The Born rule was the clearest case:

    amplitude = exp(-E / 2T);  L2-normalize;  probability = amplitude^2

is *algebraically identical* to softmax(-E / T) when the amplitudes are
real and non-negative -- verified numerically to fp32 epsilon
(5.96e-08). And with no phases, |sum_k c_k|^2 = sum_k |c_k|^2 exactly:
there are no cross-terms, so nothing can interfere. The module named
"interference" could not interfere; the "collapse" was a temperature-
scaled softmax; the "superposition" was a tensor axis.

v45 gives each hypothesis a complex amplitude

    c_k = r_k * exp(i * phi_k)

and each (hypothesis, option) compatibility a complex amplitude

    v_kn = t_kn * exp(i * theta_kn),      t_kn = exp(-E_kn / 2T)

and scores option n by the expectation of a rank-1 Hermitian projector
P_n = |v_n><v_n| in the hypothesis state |psi> = sum_k c_k |k>:

    <psi| P_n |psi>  =  | sum_k c_k v_kn |^2
                     =  sum_k |c_k|^2 |v_kn|^2                 <- classical
                        + sum_{i != j} 2 Re(c_i conj(c_j) v_in conj(v_jn))

The second line is the whole point. It is a genuine cross-term, it can
be NEGATIVE, and therefore two hypotheses can cancel each other's
support for an option. That is destructive interference, it is not
expressible as a mixture over a latent variable, and it is the one place
this project earns the word "quantum".

    P_n is Hermitian and positive semi-definite, so <psi|P_n|psi> is
    real and non-negative -- a legitimate measurement, not a metaphor.
    The classical line is exactly the decohered (diagonal) limit, which
    is what `phase_mode="classical"` computes.

Falsifiability
--------------
This is a claim, so it ships with the test. `interference_ratio` is
mean|coherent - classical| / mean(classical), reported every epoch. If
it sits at ~0, the phases learned nothing and the coherent formulation
is inert -- exactly the failure mode of every previous "quantum"
component here, and it will show up as a number instead of hiding behind
an accuracy that looks plausible. The ablation grid contains
parameter-matched `phase_mode="zero"` and `phase_mode="classical"` arms
so the effect can be attributed rather than assumed.

v46 adds the comparison those diagnostics cannot make. `interference_ratio`,
`phase_effect` and `destructive_fraction` all say the mechanism is LIVE;
none of them says it helps. `destructive_fraction` is the clearest case: with
K = 2N matched support/attack pairs and near-equal amplitudes it is a step
function of the single relative phase between the two channels
(models/full_model.py::polarity_phase). Sweeping that scalar through this
module with everything else held fixed gives 0.000 from 0 to 1.6 rad, 0.494
at 2.4, 0.994 at 2.8, and 1.000 at 3.0 and above -- and it is initialized at
pi, past the step, so it reads 1.0 for an untrained model.

The number that decides the claim is therefore `coherent_acc` vs
`classical_acc` in training/evaluate.py: the accuracy of ranking options by
`coherent` against ranking them by `classical`, the decohered limit of the
SAME amplitudes over the SAME energies in the same forward pass. If the
coherent sum does not rank options better than the mixture it replaced, the
complex amplitudes are not earning their place however live they measure.

Modes
-----
    "learned"   full complex amplitudes; phases supplied by the caller
    "zero"      coherent sum, all phases pinned to 0 -- constructive
                only. Isolates "having phases" from "summing amplitudes
                before squaring", which are two different claims.
    "classical" diagonal/decohered: P(n) = sum_k |c_k|^2 |v_kn|^2, i.e.
                a plain probability-weighted mixture over hypotheses,
                the v44 behaviour.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("learned", "zero", "classical")


class CoherentCollapse(nn.Module):

    def __init__(self, phase_mode="learned"):

        super().__init__()

        if phase_mode not in MODES:
            raise ValueError(f"phase_mode must be one of {MODES}, got {phase_mode!r}")

        self.phase_mode = phase_mode

        # Two temperatures, one per axis. Stored as logits into
        # 0.3 + softplus(.) so they stay strictly positive without a
        # clamp, and start near 1.0.
        self.hypothesis_temp = nn.Parameter(torch.tensor(0.5413))
        self.option_temp = nn.Parameter(torch.tensor(0.5413))

    def forward(
        self,
        collapse_energy,
        answer_energy,
        phase_h=None,
        phase_kn=None,
        H_mask=None,
        O_mask=None,
        hyp_temperature=None,
    ):
        """
        collapse_energy : (B, K)     fused per-hypothesis energy
        answer_energy   : (B, K, N)  per (hypothesis, option) energy
        phase_h         : (B, K)     hypothesis phases, radians
        phase_kn        : (B, K, N)  compatibility phases, radians
        H_mask          : (B, K) bool
        O_mask          : (B, N) bool
        hyp_temperature : (B, 1) optional per-sample temperature from
                          models/collapse.py, overriding the global one

        Returns log-probabilities over options. They are already
        normalized, so F.cross_entropy(log_probs, y) is exact:
        log_softmax(log p) = log p - log(sum p) = log p.
        """

        B, K, N = answer_energy.shape

        # ------------------------------------------------------------
        # Hypothesis amplitudes  r_k,  normalized so sum_k r_k^2 = 1
        # ------------------------------------------------------------

        if hyp_temperature is not None:
            t_hyp = hyp_temperature
        else:
            t_hyp = 0.3 + F.softplus(self.hypothesis_temp)

        log_r = -collapse_energy / (2.0 * t_hyp)

        if H_mask is not None:
            log_r = log_r.masked_fill(~H_mask, float("-inf"))

        log_r = log_r - log_r.max(dim=1, keepdim=True).values

        r = torch.exp(log_r)

        r = r / torch.sqrt((r ** 2).sum(dim=1, keepdim=True) + 1e-8)

        # |c_k|^2 -- the marginal weight of each hypothesis. Same object
        # v44 called `collapse_probs`, kept under that name so the
        # existing collapse diagnostics and input-ablation checks read
        # the same quantity.
        collapse_probs = r.pow(2)

        collapse_probs = collapse_probs / (
            collapse_probs.sum(dim=1, keepdim=True) + 1e-8
        )

        # ------------------------------------------------------------
        # Compatibility amplitudes  t_kn
        # ------------------------------------------------------------

        t_opt = 0.3 + F.softplus(self.option_temp)

        log_t = -answer_energy / (2.0 * t_opt)

        invalid = torch.zeros(B, K, N, dtype=torch.bool, device=log_t.device)

        if H_mask is not None:
            invalid = invalid | ~H_mask.view(B, K, 1)

        if O_mask is not None:
            invalid = invalid | ~O_mask.view(B, 1, N)

        log_t = log_t.masked_fill(invalid, float("-inf"))

        # Per-sample max subtraction over BOTH axes jointly: answer_energy
        # is clamped to +/-12 upstream and the temperature can fall to
        # 0.3, so exp(-E/2T) reaches e^20 without this.
        log_t = log_t - log_t.reshape(B, -1).max(dim=1).values.view(B, 1, 1)

        t = torch.exp(log_t)

        # ------------------------------------------------------------
        # Classical (decohered) and coherent option weights
        # ------------------------------------------------------------

        w = r.unsqueeze(-1) * t  # |c_k| * |v_kn|

        classical = (w ** 2).sum(dim=1)  # (B, N)

        # The in-phase (all phases 0) coherent sum. Reference point for
        # `phase_effect` below, and the value used directly in "zero"
        # mode.
        in_phase = w.sum(dim=1) ** 2

        if self.phase_mode == "classical":

            coherent = classical

        elif self.phase_mode == "zero" or phase_h is None:

            coherent = in_phase

        else:

            phase = phase_h.unsqueeze(-1).expand_as(w)

            if phase_kn is not None:
                phase = phase + phase_kn

            real = (w * torch.cos(phase)).sum(dim=1)
            imag = (w * torch.sin(phase)).sum(dim=1)

            coherent = real ** 2 + imag ** 2

        interference = coherent - classical

        # ------------------------------------------------------------
        # Normalize to a distribution over options
        # ------------------------------------------------------------

        weights = coherent

        if O_mask is not None:
            weights = weights.masked_fill(~O_mask, 0.0)

        weights = weights.clamp(min=1e-12)

        log_probs = torch.log(weights) - torch.log(
            weights.sum(dim=1, keepdim=True)
        )

        if O_mask is not None:
            neg_inf = torch.finfo(log_probs.dtype).min / 2
            log_probs = log_probs.masked_fill(~O_mask, neg_inf)

        # ------------------------------------------------------------
        # Diagnostics
        # ------------------------------------------------------------

        if O_mask is not None:
            om = O_mask.to(classical.dtype)
        else:
            om = torch.ones_like(classical)

        denom = om.sum().clamp(min=1.0)

        def masked_mean(t):
            return (t * om).sum() / denom

        mean_classical = masked_mean(classical)

        # How far the coherent sum is from the decohered mixture. Note
        # that this is LARGE even with all phases at zero -- a purely
        # constructive coherent sum is already a long way from a
        # mixture -- so it says "amplitudes are being summed before
        # squaring", not "the phases are doing something".
        interference_ratio = masked_mean(interference.abs()) / (
            mean_classical + 1e-12
        )

        # The two numbers that actually test the phase channel:
        #
        #   phase_effect        how much the learned phases move the
        #                       result away from the in-phase sum. Zero
        #                       means the phases are inert and this is
        #                       just constructive summation.
        #   destructive_fraction  the share of (sample, option) entries
        #                       where the cross-term is NEGATIVE, i.e.
        #                       hypotheses cancelling. It is identically
        #                       0 for phase_mode "zero" and "classical",
        #                       because a sum of non-negative terms
        #                       cannot cancel. Anything above 0 here is
        #                       destructive interference, which no
        #                       mixture over a latent variable can
        #                       express.
        phase_effect = masked_mean((coherent - in_phase).abs()) / (
            masked_mean(in_phase) + 1e-12
        )

        destructive_fraction = masked_mean((interference < 0).to(classical.dtype))

        # Relative spread of the option weights, (max - min) / mean, for
        # the coherent sum and for the decohered mixture it replaced.
        #
        # Read the obvious way round this is actively misleading, so state
        # it explicitly: contrast RISES under cancellation. When the
        # coherent sum nearly cancels, what survives is a small residual,
        # and the relative spread of a small residual is amplified. A high
        # `coherent_contrast` therefore says the phases pulled the option
        # weights apart -- NOT that they pulled the correct option away
        # from the rest. Separating those two takes `ece` and the
        # coherent_acc vs classical_acc comparison in evaluate.py.
        valid_opt = om > 0

        n_opt = om.sum(dim=1).clamp(min=1.0)

        def option_contrast(x):

            x_mean = (x * om).sum(dim=1) / n_opt

            x_max = x.masked_fill(~valid_opt, float("-inf")).max(dim=1).values

            x_min = x.masked_fill(~valid_opt, float("inf")).min(dim=1).values

            return ((x_max - x_min) / (x_mean + 1e-12)).mean()

        coherent_contrast = option_contrast(coherent).detach()

        classical_contrast = option_contrast(classical).detach()

        entropy = -(
            log_probs.exp() * log_probs.clamp(min=-30)
        ).sum(dim=1)

        return {
            "log_probs": log_probs,
            "probabilities": log_probs.exp(),
            "collapse_probs": collapse_probs,
            "amplitude": r,
            "classical": classical,
            "coherent": coherent,
            "interference": interference,
            "interference_ratio": interference_ratio,
            "phase_effect": phase_effect,
            "destructive_fraction": destructive_fraction,
            "coherent_contrast": coherent_contrast,
            "classical_contrast": classical_contrast,
            "answer_entropy": entropy.mean(),
            "hypothesis_temperature": (
                t_hyp.detach().mean() if torch.is_tensor(t_hyp) else t_hyp
            ),
            "option_temperature": t_opt.detach(),
        }

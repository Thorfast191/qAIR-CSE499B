"""
Hypothesis generator.

v45 -- asymmetric evidence
--------------------------
This is the decisive change in v45, and the reason for it is worth
stating precisely, because it is a DESIGN flaw rather than a bug and no
amount of downstream architecture could have compensated for it.

v44 asked the LLM, for each option: "state the reasoning that would make
this proposed answer correct." A competent model obliges -- for all four
options. It manufactures symmetric support. Four equally fluent
justifications contain no evidence about which option is true, so the
hypotheses were, information-theoretically, close to constant with
respect to the label. Measured on the clean v44 cache (0% fallbacks,
alignment verified by construction, question encoded):

    argmax diag(H . O)             0.2481    chance = 0.2500
    mean pairwise cos(H_k, H_j)    0.6466
    mean pairwise cos(O_k, O_j)    0.4773

The hypotheses were LESS diverse than the options they were generated
from, and their agreement with their own option carried exactly zero
information about correctness. That single fact explains every
downstream observation without needing any of the four bugs the v44
audit found: the selector ignored H because H was uninformative, the
collapse distribution was uniform because there was nothing to
discriminate, and accuracy tracked option priors because that was the
only signal present.

v45 generates TWO hypotheses per option:

    support -- why this answer would be correct   (as in v44)
    attack  -- the specific reason it is wrong

The model will happily fabricate an attack on the correct option too;
that is expected and is not a failure. The hypothesis being tested is
that an attack on a genuinely wrong option lands on a concrete
mechanism, while an attack on the correct one comes out generic or
self-contradictory -- an ASYMMETRY between the two channels. That
asymmetry is the only thing the marginalization has to work with, and
whether it exists at all is now measured directly at cache-build time
(training/dataset.py::report_quality) rather than assumed.

Alignment (v44, kept)
---------------------
One hypothesis per (question, option, mode) pair, each in its own
completion. Alignment is positional and cannot silently drift, which is
what broke pre-v44 caches where a free-form multi-hypothesis completion
was split on newlines and line k routinely described option k-1.

Option log-likelihood
---------------------
`score_options` caches the generator's own length-normalized
log-likelihood for each option. This is the direct-LLM baseline computed
once, offline -- and, more importantly, the only channel through which
the LLM's judgement reaches the trainable model without being squeezed
through the frozen 384-d sentence encoder.
"""

import re
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from config import (
    resolve_device,
    GEN_MAX_NEW_TOKENS,
    GEN_TEMPERATURE,
    GEN_MODES,
)

LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = "You are an expert scientific reasoning system."

# +1 support / -1 attack. The sign is carried through the cache and into
# the model as a per-hypothesis feature, so the selector can treat "the
# attack on option n" differently from "the support for option n"
# instead of averaging the two into mush.
POLARITY = {"support": 1.0, "attack": -1.0}


class HypothesisGenerator:

    def __init__(self, device=None):

        self.device = device or resolve_device()

        # fp16 has no efficient kernels on most CPUs (and isn't relevant
        # on MPS the same way it is on CUDA) -- only use it on an actual
        # CUDA GPU, fp32 everywhere else.
        dtype = torch.float16 if self.device == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_NAME,
            trust_remote_code=True,
        )

        # Left-padding is required for batched causal-LM generation: with
        # left padding every sample's generated continuation starts at the
        # same offset (inputs["input_ids"].shape[1]) regardless of prompt
        # length, so generate_batch() can slice all outputs the same way.
        # score_options() flips this to "right" for the duration of a
        # scoring pass and restores it -- see the note there.
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # device_map="auto" (via accelerate) targets CUDA specifically --
        # on CPU/MPS just load normally and move the whole model over.
        if self.device == "cuda":

            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_NAME,
                torch_dtype=dtype,
                device_map="auto",
                trust_remote_code=True,
            )

        else:

            self.model = AutoModelForCausalLM.from_pretrained(
                LLM_NAME,
                torch_dtype=dtype,
                trust_remote_code=True,
            ).to(self.device)

        self.model.eval()

    # ==========================================================
    # PROMPTS -- one option, one mode, at a time
    # ==========================================================

    def build_prompt(self, question, option, mode):

        if mode == "support":

            return (
                f"Question: {question}\n"
                f"Proposed answer: {option}\n\n"
                "In ONE sentence of 15-30 words, state the scientific reasoning "
                "that would make this proposed answer correct. Explain the "
                "mechanism. Do not restate the answer, do not mention other "
                "options, and do not say whether it is right or wrong.\n\n"
                "Reasoning:"
            )

        if mode == "attack":

            return (
                f"Question: {question}\n"
                f"Proposed answer: {option}\n\n"
                "In ONE sentence of 15-30 words, state the specific scientific "
                "reason this proposed answer FAILS to answer the question. Name "
                "the concrete fact, mechanism, or condition it gets wrong. Be "
                "specific rather than generic; do not restate the answer and do "
                "not mention other options.\n\n"
                "Objection:"
            )

        raise ValueError(f"unknown generation mode {mode!r}")

    def _chat(self, prompt):

        return self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )

    # ==========================================================
    # PARSER -- one hypothesis from one completion
    # ==========================================================

    def parse_one(self, text):
        """
        Returns a cleaned hypothesis string, or None if the completion
        produced nothing usable. Deliberately does NOT filter on content
        words: the pre-v44 keyword blocklist ("question", "answer",
        "generate", "hypothesis") threw away correct science.
        """

        text = text.strip()

        # Drop a leading bullet/enumerator/label if present.
        text = re.sub(r"^\s*(?:[-*•]|\(?[A-Za-z0-9]{1,3}[\.\)\:])\s*", "", text)

        # Strip surrounding markdown emphasis.
        text = re.sub(r"[*_`]+", "", text)

        # Keep the first sentence-ish chunk; the prompt asks for one
        # sentence but small models often keep going.
        text = text.split("\n")[0].strip()

        if not text:
            return None

        words = text.split()

        if len(words) < 5:
            return None

        if len(words) > 60:
            text = " ".join(words[:60])

        return text.strip()

    # ==========================================================
    # FALLBACK
    # ==========================================================

    def fallback(self, question, option, mode="support"):
        """
        Used only when the model returns nothing usable. Callers get an
        is_fallback flag alongside so these can be masked or dropped
        rather than silently treated as real hypotheses.

        Deliberately does NOT embed the option text. The pre-v44 fallback
        did (`f"{option} correctly explains ..."`), which made it
        maximally cosine-similar to its own option -- exactly the signal
        the selector keys on, so the noise looked like signal.
        """

        if mode == "attack":
            return "No specific objection to this option was produced."

        return "No supporting reasoning was produced for this option."

    # ==========================================================
    # GENERATE (single question, all its options, all modes)
    # ==========================================================

    @torch.no_grad()
    def generate(self, question, options, modes=GEN_MODES):

        out = self.generate_batch_with_flags([question], [options], modes=modes)

        return out[0]["hypotheses"]

    @torch.no_grad()
    def generate_with_flags(self, question, options, modes=GEN_MODES):

        return self.generate_batch_with_flags([question], [options], modes=modes)[0]

    # ==========================================================
    # GENERATE (batched across questions, options AND modes)
    #
    # Every (question, option, mode) triple becomes its own prompt; the
    # flat list is batched through one model.generate() call and then
    # regrouped. Alignment is positional and cannot drift.
    # ==========================================================

    @torch.no_grad()
    def generate_batch(self, questions, options_list, modes=GEN_MODES):

        return [
            item["hypotheses"]
            for item in self.generate_batch_with_flags(
                questions, options_list, modes=modes
            )
        ]

    @torch.no_grad()
    def generate_batch_with_flags(self, questions, options_list, modes=GEN_MODES):
        """
        Returns one dict per input question:

            {
              "hypotheses":  list[str]   length len(modes) * n_options,
                                         ordered mode-major:
                                         [sup_0..sup_{N-1}, att_0..att_{N-1}]
              "is_fallback": list[bool]  same length
              "polarity":    list[float] +1 support / -1 attack
              "hyp_option":  list[int]   which option each hypothesis is about
              "modes":       list[str]
            }

        Mode-major ordering matters: training/dataset.py::collate_fn
        permutes options for shuffling and reorders each mode block by
        the same permutation, which is only correct if the blocks are
        contiguous and each is in option order.
        """

        flat_prompts = []
        owners = []

        for qi, (question, options) in enumerate(zip(questions, options_list)):
            for mode in modes:
                for oi, option in enumerate(options):
                    flat_prompts.append(
                        self._chat(self.build_prompt(question, option, mode))
                    )
                    owners.append((qi, oi, mode))

        inputs = self.tokenizer(
            flat_prompts,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            temperature=GEN_TEMPERATURE,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.10,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        input_len = inputs["input_ids"].shape[1]

        results = [
            {
                "hypotheses": [],
                "is_fallback": [],
                "polarity": [],
                "hyp_option": [],
                "modes": [],
            }
            for _ in questions
        ]

        for i, (qi, oi, mode) in enumerate(owners):

            decoded = self.tokenizer.decode(
                outputs[i][input_len:],
                skip_special_tokens=True,
            )

            hyp = self.parse_one(decoded)

            if hyp is None:
                hyp = self.fallback(questions[qi], options_list[qi][oi], mode)
                is_fallback = True
            else:
                is_fallback = False

            r = results[qi]
            r["hypotheses"].append(hyp)
            r["is_fallback"].append(is_fallback)
            r["polarity"].append(POLARITY[mode])
            r["hyp_option"].append(oi)
            r["modes"].append(mode)

        return results

    # ==========================================================
    # OPTION LOG-LIKELIHOOD
    # ==========================================================

    @torch.no_grad()
    def score_options(self, questions, options_list, batch_size=32):
        """
        Length-normalized log P(option | question) under the generator
        LLM, for every option of every question.

        Returns a list of float lists, one per question.

        This is the direct-LLM baseline, computed once at cache-build
        time and stored alongside the embeddings. Two uses:

          * evaluation/baselines.py reads it straight from the cache, so
            the single most important missing number in the v44 audit
            ("does the pipeline beat the model it is built on?") costs
            nothing to report;
          * models/full_model.py can add it to the final score as an
            explicit log-prior. It is the only path by which the LLM's
            judgement reaches the answer without passing through the
            frozen 384-d encoder.

        Right padding for the duration of the call: with left padding the
        prompt/answer boundary lands at a different offset in every row,
        so the answer-token slice below would be wrong. __init__ sets
        left padding because batched *generation* requires it; both are
        correct, for different operations.
        """

        pairs = []
        owners = []

        for qi, (question, options) in enumerate(zip(questions, options_list)):
            for option in options:
                prompt = f"Question: {question}\nAnswer:"
                pairs.append((prompt, prompt + " " + option))
                owners.append(qi)

        scores = [None] * len(pairs)

        previous_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = "right"

        try:

            for start in range(0, len(pairs), batch_size):

                chunk = pairs[start:start + batch_size]

                prompt_lens = [
                    self.tokenizer(p, return_tensors="pt").input_ids.shape[1]
                    for p, _ in chunk
                ]

                enc = self.tokenizer(
                    [f for _, f in chunk],
                    return_tensors="pt",
                    padding=True,
                ).to(self.device)

                ids = enc["input_ids"]
                attn = enc["attention_mask"]

                logits = self.model(**enc).logits[:, :-1]

                targets = ids[:, 1:]

                logprobs = torch.log_softmax(logits.float(), dim=-1).gather(
                    2, targets.unsqueeze(-1)
                ).squeeze(-1)

                valid = attn[:, 1:].bool()

                for row in range(len(chunk)):

                    # Token t of `targets` is predicted from position t,
                    # so the answer tokens start at prompt_len - 1.
                    lo = max(prompt_lens[row] - 1, 0)

                    keep = valid[row].clone()
                    keep[:lo] = False

                    n = int(keep.sum())

                    scores[start + row] = (
                        (logprobs[row][keep].sum() / n).item() if n else -1e9
                    )

        finally:
            self.tokenizer.padding_side = previous_side

        grouped = [[] for _ in questions]

        for score, qi in zip(scores, owners):
            grouped[qi].append(score)

        return grouped

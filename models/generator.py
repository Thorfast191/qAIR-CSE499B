"""
Hypothesis generator.

Rewritten after the audit. The previous design asked the LLM for all N
hypotheses in ONE completion and then split the reply on newlines. Three
things went wrong with that, measured on the shipped cache:

1. `max_new_tokens=96` could not hold 4 hypotheses of the requested
   15-25 words (~80-130 tokens). Generation truncated and the tail
   hypotheses never arrived.

2. The line parser rejected any line containing "question", "answer",
   "generate" or "hypothesis" -- words that appear constantly in
   legitimate science explanations -- discarding valid output.

3. Whatever was missing got silently replaced by a template fallback,
   `f"{option} correctly explains the {concept} in the question."`,
   which embeds the option text verbatim. 33.1% of train and 33.6% of
   validation hypotheses were this template; 315 train samples were
   100% template. Those samples are provably uninformative: every option
   receives an identically-structured string.

Worse than the rate was the ALIGNMENT. Splitting a free-form completion
on newlines does not guarantee that line k describes option k. Real
example from the shipped cache -- the model emitted "**option**" and
"explanation" as separate lines, shifting the whole correspondence:

    [0] OPT 'The refrigerator door is smooth.'
        HYP '**The refrigerator door is smooth**'          <- echo of opt 0
    [1] OPT 'The refrigerator door contains iron.'  (correct)
        HYP 'Smooth surfaces tend to attract magnetic...'  <- about opt 0
    [2] OPT 'The refrigerator door is a good conductor.'
        HYP '**The refrigerator door contains iron**'      <- echo of opt 1

The whole architecture assumes hypothesis k belongs to option k
(collate_fn even permutes H and O jointly to preserve it). Zero-shot
argmax diag(H . O) on that cache scored 0.2670 -- chance.

The fix is structural, not a better parser: generate ONE hypothesis per
(question, option) pair in its own completion. Alignment is then
guaranteed by construction and cannot silently drift. Fallbacks are
still possible but are now flagged (`is_fallback`) rather than silently
mixed in, so downstream code can mask or drop them.
"""

import re
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from config import resolve_device, GEN_MAX_NEW_TOKENS, GEN_TEMPERATURE

LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

SYSTEM_PROMPT = "You are an expert scientific reasoning system."


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
    # PROMPT -- one option at a time
    # ==========================================================

    def build_prompt(self, question, option):

        return (
            f"Question: {question}\n"
            f"Proposed answer: {option}\n\n"
            "In ONE sentence of 15-30 words, state the scientific reasoning that "
            "would make this proposed answer correct. Explain the mechanism. "
            "Do not restate the answer, do not mention other options, and do not "
            "say whether it is right or wrong.\n\n"
            "Reasoning:"
        )

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
        words: the previous keyword blocklist ("question", "answer",
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

    def fallback(self, question, option):
        """
        Used only when the model returns nothing usable for this option.
        Callers get an is_fallback flag alongside so these can be masked
        or dropped rather than silently treated as real hypotheses.

        Deliberately does NOT embed the option text. The old fallback
        did, which made it maximally cosine-similar to its own option --
        exactly the signal the selector keys on, so the noise looked like
        signal.
        """

        return "No supporting reasoning was produced for this option."

    # ==========================================================
    # GENERATE (single question, all its options)
    # ==========================================================

    @torch.no_grad()
    def generate(self, question, options):

        hyps, _ = self.generate_with_flags(question, options)

        return hyps

    @torch.no_grad()
    def generate_with_flags(self, question, options):

        results = self.generate_batch_with_flags([question], [options])

        return results[0]

    # ==========================================================
    # GENERATE (batched across questions AND options)
    #
    # Every (question, option) pair becomes its own prompt; the flat
    # list is batched through one model.generate() call and then
    # regrouped. Alignment is positional and cannot drift.
    # ==========================================================

    @torch.no_grad()
    def generate_batch(self, questions, options_list):

        return [
            hyps
            for hyps, _ in self.generate_batch_with_flags(questions, options_list)
        ]

    @torch.no_grad()
    def generate_batch_with_flags(self, questions, options_list):
        """
        Returns a list of (hypotheses, is_fallback_flags) tuples, one per
        input question, with len(hypotheses) == len(options).
        """

        flat_prompts = []
        owners = []

        for qi, (question, options) in enumerate(zip(questions, options_list)):
            for option in options:
                flat_prompts.append(self._chat(self.build_prompt(question, option)))
                owners.append((qi, option))

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

        results = [([], []) for _ in questions]

        for i, (qi, option) in enumerate(owners):

            decoded = self.tokenizer.decode(
                outputs[i][input_len:],
                skip_special_tokens=True,
            )

            hyp = self.parse_one(decoded)

            if hyp is None:
                hyp = self.fallback(questions[qi], option)
                is_fallback = True
            else:
                is_fallback = False

            results[qi][0].append(hyp)
            results[qi][1].append(is_fallback)

        return results

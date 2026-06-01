from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

import torch


LLM_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


class HypothesisGenerator:

    def __init__(self, device="cuda"):

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_NAME,
            trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        self.model.eval()

    # ========================================================
    # PROMPT
    # ========================================================

    def build_prompt(
        self,
        question,
        options
    ):

        option_block = "\n".join(
            [
                f"{chr(65+i)}. {o}"
                for i, o in enumerate(options)
            ]
        )

        return f"""
Question:
{question}

Options:
{option_block}

Generate EXACTLY four hypotheses.

1. Causal Hypothesis
2. Contradictory Hypothesis
3. Elimination Hypothesis
4. Counterfactual Hypothesis

Requirements:
- One hypothesis per line
- Keep each hypothesis under 25 words
- Do not explain
- Do not number the output
"""

    # ========================================================
    # GENERATE
    # ========================================================

    @torch.no_grad()
    def generate(
        self,
        question,
        options
    ):

        prompt = self.build_prompt(
            question,
            options
        )

        messages = [

            {
                "role": "system",
                "content":
                (
                    "You are a reasoning engine that "
                    "creates diverse hypotheses."
                )
            },

            {
                "role": "user",
                "content": prompt
            }

        ]

        text_input = (
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )

        inp = self.tokenizer(
            text_input,
            return_tensors="pt"
        ).to(self.device)

        out = self.model.generate(

            **inp,

            max_new_tokens=128,

            temperature=0.7,

            top_p=0.9,

            do_sample=True,

            pad_token_id=self.tokenizer.eos_token_id

        )

        # ====================================================
        # ONLY DECODE NEW TOKENS
        # ====================================================

        generated_tokens = out[0][
            inp["input_ids"].shape[1]:
        ]

        text = self.tokenizer.decode(
            generated_tokens,
            skip_special_tokens=True
        )

        # ====================================================
        # PARSE
        # ====================================================

        hypotheses = []

        for line in text.split("\n"):

            line = line.strip()

            if not line:
                continue

            if len(line) < 10:
                continue

            if line.lower().startswith(
                ("question", "options")
            ):
                continue

            hypotheses.append(line)

        # ====================================================
        # CLEANUP
        # ====================================================

        cleaned = []

        for h in hypotheses:

            h = h.lstrip(
                "1234567890.-) "
            )

            h = h.strip()

            if len(h) > 5:

                cleaned.append(h)

        hypotheses = cleaned[:4]

        # ====================================================
        # FALLBACKS
        # ====================================================

        fallback_templates = [

            f"The evidence supports {options[0]}.",

            f"A contradiction emerges if {options[min(1, len(options)-1)]} is correct.",

            f"Elimination favors {options[min(2, len(options)-1)]}.",

            f"A counterfactual perspective suggests {options[min(3, len(options)-1)]}."
        ]

        while len(hypotheses) < 4:

            hypotheses.append(
                fallback_templates[
                    len(hypotheses)
                ]
            )

        return hypotheses[:4]
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
            LLM_NAME
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def build_prompt(
        self,
        question,
        options
    ):

        option_block = "\n".join([
            f"{chr(65+i)}. {o}"
            for i, o in enumerate(options)
        ])

        return f"""
Question:
{question}

Options:
{option_block}

Generate exactly four hypotheses.

1. Causal Hypothesis
2. Contradictory Hypothesis
3. Elimination Hypothesis
4. Counterfactual Hypothesis

Return one hypothesis per line.
"""

    def generate(
        self,
        question,
        options
    ):

        prompt = self.build_prompt(
            question,
            options
        )

        inp = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():

            out = self.model.generate(
                **inp,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

        text = self.tokenizer.decode(
            out[0],
            skip_special_tokens=True
        )

        lines = []

        for line in text.split("\n"):

            line = line.strip()

            if len(line) > 20:

                lines.append(line)

        lines = lines[:4]

        while len(lines) < 4:

            lines.append(
                "Fallback hypothesis."
            )

        return lines
import re
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class HypothesisGenerator:

    def __init__(self, device="cuda"):

        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_NAME,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()

    # ==========================================================
    # PROMPT
    # ==========================================================

    def build_prompt(self, question, options):

        option_block = "\n".join(
            [
                f"{chr(65+i)}. {opt}"
                for i, opt in enumerate(options)
            ]
        )

        return f"""
You are an expert scientific reasoning engine.

Question:
{question}

Candidate Answers:
{option_block}

Task:

For EACH candidate answer, assume it is correct and write ONE
short hypothesis supporting that answer.

Requirements

- Produce EXACTLY {len(options)} hypotheses.
- One hypothesis per answer.
- Keep each hypothesis under 20 words.
- Make every hypothesis plausible.
- Do NOT explain.
- Do NOT number.
- Output only the hypotheses.
"""

    # ==========================================================
    # CLEAN PARSER
    # ==========================================================

    def parse_output(self, text, expected):

        hypotheses = []

        for line in text.split("\n"):

            line = line.strip()

            if not line:
                continue

            # Remove bullets / numbering only (e.g. "1. ", "A) ", "- ")
            # NOTE: previous version used [A-Za-z0-9...]+ which greedily
            # eats real words at the start of the sentence, not just the
            # bullet marker. Cap the marker to <=3 leading chars.
            line = re.sub(
                r"^[A-Za-z0-9]{0,3}[\.\)\:\-\•\*]\s*",
                "",
                line,
            )

            line = line.strip()

            if len(line) < 8:
                continue

            hypotheses.append(line)

        return hypotheses[:expected]

    # ==========================================================
    # FALLBACK
    # ==========================================================

    def fallback(self, question, options):

        hyps = []

        for option in options:

            hyps.append(
                f"If '{option}' is correct, then it best explains the question."
            )

        return hyps

    # ==========================================================
    # GENERATE
    # ==========================================================

    @torch.no_grad()
    def generate(self, question, options):

        prompt = self.build_prompt(
            question,
            options,
        )

        messages = [
            {
                "role": "system",
                "content":
                    "You generate concise scientific hypotheses.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=96,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.10,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        generated = outputs[0][inputs["input_ids"].shape[1]:]

        decoded = self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        )

        hypotheses = self.parse_output(
            decoded,
            len(options),
        )

        if len(hypotheses) < len(options):

            fallback = self.fallback(
                question,
                options,
            )

            while len(hypotheses) < len(options):

                hypotheses.append(
                    fallback[len(hypotheses)]
                )

        return hypotheses[:len(options)]
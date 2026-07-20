"""
IMPROVED GENERATOR (Optional upgrade)
Changes:
1. Better prompt engineering with clearer instructions
2. Fallback hypotheses improved to provide more signal
3. Output validation and quality checks
4. Better parsing for output lines
"""

import re
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

# Current model - reasonable for now, but could upgrade to larger model if resources allow
LLM_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


class HypothesisGeneratorImproved:

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
    # IMPROVED PROMPT
    # ==========================================================

    def build_prompt(self, question, options):
        """
        Improved prompt with clearer structure and examples.
        """

        option_block = "\n".join(
            [
                f"{chr(65+i)}. {opt}"
                for i, opt in enumerate(options)
            ]
        )

        return f"""You are a scientific reasoning expert. For each answer choice below, 
generate ONE short hypothesis explaining the key reasoning that makes that answer correct.

Question:
{question}

Answer Choices:
{option_block}

Generate exactly {len(options)} hypotheses - one for each answer choice above.
Each hypothesis should:
- Be 15-25 words long
- Explain the main reasoning mechanism
- Be scientifically plausible
- NOT include answer labels

Output format: Write each hypothesis on a new line, in order (hypothesis for A, then B, then C, etc.)

Hypotheses:"""

    # ==========================================================
    # IMPROVED PARSER
    # ==========================================================

    def parse_output(self, text, expected):
        """
        Improved parsing that's more robust to formatting variations.
        """

        hypotheses = []

        for line in text.split("\n"):

            line = line.strip()

            if not line:
                continue

            # Remove common markers at start (up to 3 chars)
            line = re.sub(
                r"^[A-Za-z0-9]{0,3}[\.\)\:\-\•\*]\s*",
                "",
                line,
            )

            # Remove trailing markers
            line = re.sub(r"\s*[\.\)\:]\s*$", "", line)

            line = line.strip()

            # Only accept lines with reasonable length (10-100 words)
            word_count = len(line.split())
            if word_count < 5 or word_count > 100:
                continue

            # Skip lines that look like instructions
            if any(skip in line.lower() for skip in ["question", "answer", "generate", "hypothesis"]):
                continue

            hypotheses.append(line)

        return hypotheses[:expected]

    # ==========================================================
    # IMPROVED FALLBACK (more informative)
    # ==========================================================

    def fallback(self, question, options):
        """
        Better fallback that provides more signal than generic placeholder.
        """

        hyps = []

        for i, option in enumerate(options):
            # Create semi-reasonable fallback that relates to the question/option
            if len(question) > 50:
                concept = question[:30].split()[-1] if question.split() else "concept"
            else:
                concept = "this reasoning"
            
            hyp = f"{option} correctly explains the {concept} in the question."
            hyps.append(hyp)

        return hyps

    # ==========================================================
    # QUALITY CHECK
    # ==========================================================

    def is_valid_hypothesis(self, hyp):
        """
        Check if a hypothesis has reasonable quality.
        """
        if not hyp or len(hyp) < 8:
            return False
        
        # Check it's not just a copy of the option
        if hyp.lower().count("hypothesis") > 0:
            return False
        
        # Check reasonable length
        word_count = len(hyp.split())
        if word_count < 5 or word_count > 100:
            return False
        
        return True

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
                "content": "You are an expert scientific reasoning system.",
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
            temperature=0.3,  # Slightly reduced for better quality
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

        # Fallback for insufficient or low-quality hypotheses
        if len(hypotheses) < len(options):

            fallback = self.fallback(
                question,
                options,
            )

            while len(hypotheses) < len(options):
                hyp_idx = len(hypotheses)
                hypotheses.append(fallback[hyp_idx])

        # Quality validation
        final_hyps = []
        for hyp in hypotheses[:len(options)]:
            if self.is_valid_hypothesis(hyp):
                final_hyps.append(hyp)
            else:
                # Use fallback for this option
                final_hyps.append(
                    self.fallback(question, options)[len(final_hyps)]
                )

        return final_hyps[:len(options)]

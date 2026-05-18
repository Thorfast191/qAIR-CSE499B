from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

LLM_NAME = "google/gemma-2-2b-it"


class HypothesisGenerator:

    def __init__(self, device="cuda"):
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(LLM_NAME)

        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    def build_prompt(self, question, options):

        return f"""
Question:
{question}

Options:
A. {options[0]}
B. {options[1]}
C. {options[2]}
D. {options[3]}

Generate:
1. causal hypothesis
2. contradictory hypothesis
3. elimination hypothesis
4. counterfactual hypothesis

Return concise reasoning.
"""

    def generate(self, question, options):

        prompt = self.build_prompt(question, options)

        inp = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inp,
                max_new_tokens=180,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )

        txt = self.tokenizer.decode(out[0], skip_special_tokens=True)

        hyps = []

        for line in txt.split("\n"):
            line = line.strip()
            if len(line) > 20:
                hyps.append(line)

        hyps = hyps[:4]

        while len(hyps) < 4:
            hyps.append("Fallback hypothesis")

        return hyps
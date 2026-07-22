import os
import torch

from config import EMBEDDING_DIM
from models.generator import HypothesisGenerator
from models.encoder import HypothesisEncoder
from models.full_model import QAIRvNext

TEST_SAMPLES = [
    {
        "question": "What gas do plants absorb from the atmosphere?",
        "options": ["Oxygen", "Nitrogen", "Carbon dioxide", "Hydrogen"],
        "answer": 2,
    },
    {
        "question": "Why does ice float on water?",
        "options": [
            "It is heavier than water",
            "It is less dense than water",
            "It absorbs heat",
            "It contains oxygen",
        ],
        "answer": 1,
    },
    {
        "question": "Why do shadows become shorter at noon?",
        "options": [
            "The Sun is highest in the sky",
            "The Earth stops rotating",
            "Light becomes weaker",
            "The atmosphere shrinks",
        ],
        "answer": 0,
    },
    {
        "question": "Why do seatbelts reduce injuries during accidents?",
        "options": [
            "They increase speed",
            "They reduce momentum",
            "They prevent sudden body movement",
            "They reduce gravity",
        ],
        "answer": 2,
    },
    {
        "question": "What would most likely happen if gravity on Earth doubled?",
        "options": [
            "People would weigh more",
            "Days would become longer",
            "The Sun would become brighter",
            "Water would freeze instantly",
        ],
        "answer": 0,
    },
    {
        "question": "Why do metal railway tracks have small gaps between sections?",
        "options": [
            "To allow expansion during heating",
            "To reduce weight",
            "To increase friction",
            "To improve color",
        ],
        "answer": 0,
    },
    {
        "question": "If the Earth suddenly stopped rotating, what would happen first?",
        "options": [
            "The Moon would disappear",
            "Objects would continue moving eastward",
            "Gravity would stop",
            "The Sun would explode",
        ],
        "answer": 1,
    },
    {
        "question": "Which planet is known as the Red Planet?",
        "options": [
            "Venus",
            "Jupiter",
            "Mars",
            "Saturn",
        ],
        "answer": 2,
    },
    {
        "question": "A doctor gives you four medicines and says one is harmful. What is the safest strategy?",
        "options": [
            "Take all medicines",
            "Avoid all medicines",
            "Identify and eliminate the harmful one",
            "Choose randomly",
        ],
        "answer": 2,
    },
    {
        "question": "A rooster lays an egg on the roof of a house. Which side does the egg fall from?",
        "options": [
            "Left",
            "Right",
            "Front",
            "Roosters do not lay eggs",
        ],
        "answer": 3,
    },
]

def load_ablation_model(ckpt_dir, name, cfg, device, n_qubits=12):
    """
    Loads a specific named ablation checkpoint (e.g. "A4_full_hybrid")
    with the EXACT config it was trained with, so persistent_steps /
    use_quantum / use_validator can never silently mismatch.
    """

    model = QAIRvNext(
        dim=EMBEDDING_DIM,
        use_quantum=cfg["use_quantum"],
        use_validator=cfg["use_validator"],
        persistent_steps=cfg["persistent_steps"],
        n_qubits=n_qubits,
    )

    ckpt_path = os.path.join(ckpt_dir, f"{name}_best.pt")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(ckpt["model"])

    model = model.to(device)
    model.eval()

    print(f"[LOADED] {name} from {ckpt_path} (best_acc={ckpt.get('best_acc', '?')})")

    return model


def run_sample_evaluation(model, device):

    model.eval()

    generator = HypothesisGenerator(device=device)
    encoder = HypothesisEncoder(device=device)

    correct = 0

    print("\n" + "=" * 70)
    print("qAIR REASONING EVALUATION")
    print("=" * 70)

    for idx, sample in enumerate(TEST_SAMPLES):

        question = sample["question"]
        options = sample["options"]
        answer = sample["answer"]

        hypotheses = generator.generate(question, options)

        H = encoder.encode(hypotheses).unsqueeze(0)
        O = encoder.encode(options).unsqueeze(0)

        with torch.no_grad():
            outputs = model(H.to(device), O.to(device))

        pred = outputs["scores"].argmax(dim=-1).item()

        collapse = outputs["collapse_probs"][0].detach().cpu().numpy()

        is_correct = pred == answer

        if is_correct:
            correct += 1

        print("\n" + "-" * 70)
        print(f"Sample {idx+1}")
        print("-" * 70)
        print(f"Question:\n{question}\n")
        print("Options:")
        for i, op in enumerate(options):
            print(f"{i}. {op}")
        print("\nHypotheses:")
        for i, h in enumerate(hypotheses):
            print(f"{i+1}. {h}")
        print()
        print("Collapse Probabilities:")
        print(collapse)
        print()
        print("Prediction:")
        print(options[pred])
        print()
        print("Ground Truth:")
        print(options[answer])
        print()
        print("Correct:")
        print("YES" if is_correct else "NO")

    acc = correct / len(TEST_SAMPLES)

    print("\n" + "=" * 70)
    print(f"Final Accuracy: {acc:.4f}")
    print(f"{correct}/{len(TEST_SAMPLES)}")
    print("=" * 70)

    return acc
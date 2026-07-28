import os
import torch

from config import EMBEDDING_DIM, MODEL_DIM, N_QUBITS, GEN_MODES
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

def load_ablation_model(ckpt_dir, name, cfg, device, n_qubits=N_QUBITS):
    """
    Loads a specific named ablation checkpoint (e.g. "A3_persistent")
    with the EXACT config it was trained with, so persistent_steps /
    backend / phase_mode can never silently mismatch.
    """

    from training.ablations import resolve_config

    cfg = resolve_config(cfg)

    model = QAIRvNext(
        dim=EMBEDDING_DIM,
        model_dim=cfg.get("model_dim", MODEL_DIM),
        use_quantum=cfg["use_quantum"],
        use_validator=cfg["use_validator"],
        persistent_steps=cfg["persistent_steps"],
        n_qubits=n_qubits,
        backend=cfg.get("backend"),
        use_question=cfg["use_question"],
        phase_mode=cfg["phase_mode"],
        phase_source=cfg["phase_source"],
        mixing=cfg["mixing"],
        use_llm_prior=cfg["use_llm_prior"],
        use_attack=cfg["use_attack"],
    )

    ckpt_path = os.path.join(ckpt_dir, f"{name}_best.pt")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    model.load_state_dict(ckpt["model"])

    model = model.to(device)
    model.eval()

    print(f"[LOADED] {name} from {ckpt_path} (best_acc={ckpt.get('best_acc', '?')})")

    return model


def run_sample_evaluation(model, device, generator=None, encoder=None,
                          modes=GEN_MODES):
    """
    Runs the live pipeline end to end on hand-written questions: generate
    support + attack per option, encode, and push everything the trained
    model expects (polarity, alignment, cached-equivalent LLM
    log-likelihood) through a single forward pass.

    n=10, so this is a smoke test of the live path, not a metric. Its
    value is qualitative: the printed support/attack pairs show whether
    the generator is producing evidence or just fluent text, which no
    aggregate accuracy will tell you.
    """

    model.eval()

    generator = generator or HypothesisGenerator(device=device)
    encoder = encoder or HypothesisEncoder(device=device)

    correct = 0

    print("\n" + "=" * 70)
    print("qAIR REASONING EVALUATION")
    print("=" * 70)

    for idx, sample in enumerate(TEST_SAMPLES):

        question = sample["question"]
        options = sample["options"]
        answer = sample["answer"]

        item = generator.generate_batch_with_flags(
            [question], [options], modes=modes
        )[0]

        hypotheses = item["hypotheses"]

        llm_scores = generator.score_options([question], [options])[0]

        H = encoder.encode(hypotheses).unsqueeze(0).to(device)
        O = encoder.encode(options).unsqueeze(0).to(device)
        Q = encoder.encode([question]).to(device)

        K, N = H.shape[1], O.shape[1]

        polarity = torch.tensor(
            item["polarity"], dtype=H.dtype, device=device
        ).unsqueeze(0)

        align = torch.zeros(1, K, N, dtype=H.dtype, device=device)
        align[0, torch.arange(K), torch.tensor(item["hyp_option"])] = 1.0

        llm_logprob = torch.tensor(
            llm_scores, dtype=H.dtype, device=device
        ).unsqueeze(0)

        with torch.no_grad():
            outputs = model(
                H, O, Q=Q,
                polarity=polarity, align=align, llm_logprob=llm_logprob,
            )

        pred = outputs["scores"].argmax(dim=-1).item()

        collapse = outputs["collapse_probs"][0].detach().cpu().numpy()

        is_correct = pred == answer

        if is_correct:
            correct += 1

        print("\n" + "-" * 70)
        print(f"Sample {idx+1}")
        print("-" * 70)
        print(f"Question:\n{question}\n")

        for i, op in enumerate(options):
            mark = "*" if i == answer else " "
            print(f" {mark}[{i}] {op}   (llm logp {llm_scores[i]:+.3f})")
            for m_i, mode in enumerate(modes):
                print(f"       {mode:<8s}: {hypotheses[m_i * len(options) + i]}")

        print()
        print("Collapse probabilities (per hypothesis):")
        print(collapse)
        print(f"Interference ratio: {float(outputs['interference_ratio']):.4f}")
        print()
        print(f"Prediction   : {options[pred]}")
        print(f"Ground truth : {options[answer]}")
        print("Correct      :", "YES" if is_correct else "NO")

    acc = correct / len(TEST_SAMPLES)

    print("\n" + "=" * 70)
    print(f"Final Accuracy: {acc:.4f}")
    print(f"{correct}/{len(TEST_SAMPLES)}")
    print("=" * 70)

    return acc
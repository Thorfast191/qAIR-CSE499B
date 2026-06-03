import torch

from models.generator import (
    HypothesisGenerator
)

from models.encoder import (
    HypothesisEncoder
)

from tests.reasoning_samples import (
    TEST_SAMPLES
)


def run_sample_evaluation(
    model,
    device
):

    model.eval()

    generator = HypothesisGenerator(
        device=device
    )

    encoder = HypothesisEncoder(
        device=device
    )

    correct = 0

    print(
        "\n" + "=" * 70
    )

    print(
        "qAIR REASONING EVALUATION"
    )

    print(
        "=" * 70
    )

    for idx, sample in enumerate(
        TEST_SAMPLES
    ):

        question = sample[
            "question"
        ]

        options = sample[
            "options"
        ]

        answer = sample[
            "answer"
        ]

        hypotheses = generator.generate(
            question,
            options
        )

        H = encoder.encode(
            hypotheses
        ).unsqueeze(0)

        O = encoder.encode(
            options
        ).unsqueeze(0)

        with torch.no_grad():

            outputs = model(

                H.to(device),

                O.to(device)

            )

        pred = outputs[
            "scores"
        ].argmax(
            dim=-1
        ).item()

        collapse = (

            outputs[
                "collapse_probs"
            ][0]

            .detach()

            .cpu()

            .numpy()

        )

        is_correct = (
            pred == answer
        )

        if is_correct:
            correct += 1

        print(
            "\n" + "-" * 70
        )

        print(
            f"Sample {idx+1}"
        )

        print(
            "-" * 70
        )

        print(
            f"Question:\n{question}\n"
        )

        print(
            "Options:"
        )

        for i, op in enumerate(
            options
        ):

            print(
                f"{i}. {op}"
            )

        print(
            "\nHypotheses:"
        )

        for i, h in enumerate(
            hypotheses
        ):

            print(
                f"{i+1}. {h}"
            )

        print()

        print(
            "Collapse Probabilities:"
        )

        print(
            collapse
        )

        print()

        print(
            "Prediction:"
        )

        print(
            options[pred]
        )

        print()

        print(
            "Ground Truth:"
        )

        print(
            options[answer]
        )

        print()

        print(
            "Correct:"
        )

        print(
            "YES"
            if is_correct
            else "NO"
        )

    acc = (
        correct
        /
        len(TEST_SAMPLES)
    )

    print(
        "\n" + "=" * 70
    )

    print(
        f"Final Accuracy: "
        f"{acc:.4f}"
    )

    print(
        f"{correct}/"
        f"{len(TEST_SAMPLES)}"
    )

    print(
        "=" * 70
    )

    return acc
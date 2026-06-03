import torch

from torch.utils.data import (
    DataLoader
)

from training.dataset import (
    QAIRDataset,
    collate_fn,
    DIM
)

from training.train import (
    Trainer
)

from models.full_model import (
    QAIRvNext
)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


ABLATIONS = {

    "A1_baseline": {

        "use_quantum": False,

        "use_validator": False,

        "persistent_steps": 1

    },

    "A8_full_hybrid": {

        "use_quantum": True,

        "use_validator": True,

        "persistent_steps": 3

    },

    "A9_validator": {

        "use_quantum": False,

        "use_validator": True,

        "persistent_steps": 3

    },

    "A10_persistent": {

        "use_quantum": True,

        "use_validator": True,

        "persistent_steps": 5

    }

}


def run_ablation_suite(
    cache_dir,
    ckpt_dir
):

    train_ds = QAIRDataset(

        split="train",

        max_samples=500,

        cache_dir=cache_dir

    )

    val_ds = QAIRDataset(

        split="validation",

        max_samples=100,

        cache_dir=cache_dir

    )

    train_loader = DataLoader(

        train_ds,

        batch_size=8,

        shuffle=True,

        collate_fn=collate_fn

    )

    val_loader = DataLoader(

        val_ds,

        batch_size=8,

        shuffle=False,

        collate_fn=collate_fn

    )

    results = {}

    for name, cfg in ABLATIONS.items():

        print(
            "\n" + "=" * 60
        )

        print(
            f"Running {name}"
        )

        model = QAIRvNext(

            dim=DIM,

            use_quantum=cfg[
                "use_quantum"
            ],

            use_validator=cfg[
                "use_validator"
            ],

            persistent_steps=cfg[
                "persistent_steps"
            ]

        ).to(device)

        trainer = Trainer(

            model=model,

            train_loader=train_loader,

            val_loader=val_loader,

            device=device,

            ckpt_dir=ckpt_dir,

            name=name

        )

        trainer.train(
            epochs=5
        )

        results[name] = {

            "use_quantum":
                cfg["use_quantum"],

            "use_validator":
                cfg["use_validator"],

            "persistent_steps":
                cfg["persistent_steps"]

        }

    print(
        "\nAblation Suite Complete."
    )

    return results
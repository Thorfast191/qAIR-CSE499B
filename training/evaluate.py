import torch

from tqdm.auto import tqdm


@torch.no_grad()
def evaluate(
    model,
    loader,
    device
):

    model.eval()

    correct = 0
    total = 0

    spreads = []
    entropies = []
    diversities = []

    for batch in tqdm(
        loader,
        desc="Validation",
        leave=False
    ):

        H = batch["H"].to(device)
        O = batch["O"].to(device)
        y = batch["y"].to(device)

        out = model(H, O)

        pred = out["scores"].argmax(
            dim=1
        )

        correct += (
            pred == y
        ).sum().item()

        total += y.size(0)

        spreads.append(
            out["hypothesis_energy"]
            .var(dim=1)
            .mean()
            .item()
        )

        entropies.append(
            out["entropy"].item()
        )

        diversities.append(
            out["diversity"].item()
        )

    return {

        "acc":
            correct / total,

        "spread":
            sum(spreads) /
            len(spreads),

        "entropy":
            sum(entropies) /
            len(entropies),

        "diversity":
            sum(diversities) /
            len(diversities)

    }
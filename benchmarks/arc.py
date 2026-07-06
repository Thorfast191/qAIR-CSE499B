from datasets import load_dataset, concatenate_datasets, DatasetDict


def load_arc():
    """
    Loads and merges ARC-Challenge + ARC-Easy.

    ARC-Challenge alone only has 1,119 train / 299 validation / 1,172 test
    examples -- nowhere near enough for this model. Combining with ARC-Easy
    roughly triples the pool (adds ~2,251 train / 570 validation / 2,376 test),
    same format, same domain (grade-school science), so it merges cleanly.
    """

    challenge = load_dataset("ai2_arc", "ARC-Challenge")
    easy = load_dataset("ai2_arc", "ARC-Easy")

    merged = DatasetDict()

    for split in challenge.keys():

        if split in easy:
            merged[split] = concatenate_datasets(
                [challenge[split], easy[split]]
            )
        else:
            merged[split] = challenge[split]

    return merged
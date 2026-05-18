from datasets import load_dataset


def load_arc():

    ds = load_dataset("ai2_arc", "ARC-Challenge")

    return ds
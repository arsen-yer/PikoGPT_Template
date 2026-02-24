from datasets import load_dataset


def load_openwebtext_subset(subset_size=10000, streaming=False):
    """
    Load OpenWebText with configurable subset size.

    Args:
        subset_size (int): Number of samples to load
        streaming (bool): Whether to stream dataset (recommended for large size)

    Returns:
        List of text samples
    """

    print("Loading OpenWebText...")

    dataset = load_dataset(
        "Skylion007/openwebtext",
        split="train",
        streaming=streaming,
    )

    if streaming:
        samples = []
        for i, sample in enumerate(dataset):
            if i >= subset_size:
                break
            samples.append(sample["text"])
        return samples

    dataset = dataset.select(range(subset_size))
    return dataset["text"]


if __name__ == "__main__":
    subset = load_openwebtext_subset(subset_size=5000, streaming=True)
    print(f"Loaded {len(subset)} samples.")

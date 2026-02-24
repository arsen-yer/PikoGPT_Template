from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from datasets import load_dataset

from src.io_utils import write_jsonl, write_metadata
from src.text_cleaning import (
    clean_text,
    looks_english,
    normalize_text,
)


def load_owt_subset(
    name: str,
    split: str,
    num_samples: int,
    seed: int,
    streaming: bool = True,
) -> list[str]:
    """Load a subset of OpenWebText texts."""
    if streaming:
        dataset = load_dataset(name, split=split, streaming=True)
        texts: list[str] = []
        for item in dataset:
            text = item.get("text")
            if text:
                texts.append(text)
            if len(texts) >= num_samples:
                break
        return texts

    dataset = load_dataset(name, split=split)
    dataset = dataset.shuffle(seed=seed).select(range(num_samples))
    return [row["text"] for row in dataset if "text" in row]


def preprocess_openwebtext(
    output_path: Path,
    subset: str = "train",
    num_samples: int = 5000,
    seed: int = 42,
    streaming: bool = True,
    metadata_args: dict[str, Any] | None = None,
) -> dict[str, int]:
    """Preprocess OpenWebText with lightweight heuristics.

    Note: English filtering is heuristic and will miss some English
    content while retaining some non-English text.
    """
    texts = load_owt_subset(
        name="openwebtext",
        split=subset,
        num_samples=num_samples,
        seed=seed,
        streaming=streaming,
    )

    cleaned: list[str] = []
    for text in texts:
        cleaned_text = normalize_text(clean_text(text))
        if not cleaned_text:
            continue
        if not looks_english(cleaned_text):
            continue
        cleaned.append(cleaned_text)

    items = [{"text": item} for item in cleaned]
    write_jsonl(output_path, items)

    stats = {
        "subset_loaded": len(texts),
        "after_cleaning": len(cleaned),
    }

    metadata = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "args": metadata_args or {},
        "seed": seed,
        "counts": stats,
    }
    metadata_path = output_path.with_suffix(output_path.suffix + ".meta.json")
    write_metadata(metadata_path, metadata)
    return stats

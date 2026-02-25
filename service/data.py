from __future__ import annotations



from pathlib import Path
import csv
import html
import json
import re
import zipfile

from datasets import load_dataset


def load_csv(path: str | Path) -> list[dict[str, str]]:
    
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path.as_posix()}")

    with file_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def preprocess_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Trim string fields for simple row-level cleanup.
    """
    cleaned: list[dict[str, str]] = []
    for row in rows:
        cleaned.append(
            {
                key: value.strip() if isinstance(value, str) else value
                for key, value in row.items()
            }
        )
    return cleaned


def load_hf_dataset(name: str, split: str) -> list[dict[str, str]]:
    """Load a HuggingFace dataset split into memory.

    Args:
        name: Dataset identifier on HuggingFace.
        split: Split name, e.g., "train".

    Returns:
        A list of dataset rows as dictionaries.
    """
    dataset = load_dataset(name, split=split)
    return list(dataset)


def load_owt_subset(
    name: str,
    split: str,
    subset_size: int,
    seed: int,
    streaming: bool = False,
) -> list[str]:
    """Load a subset of OpenWebText texts.

    Streaming reads examples sequentially without downloading the full dataset.
    Non-streaming supports shuffling for a more random subset but downloads
    dataset metadata and cached data locally.


    Returns:
        A list of text strings from the dataset.
    """
    if streaming:
        dataset = load_dataset(name, split=split, streaming=True)
        texts: list[str] = []
        for item in dataset:
            text = item.get("text")
            if text:
                texts.append(text)
            if len(texts) >= subset_size:
                break
        return texts

    dataset = load_dataset(name, split=split)
    dataset = dataset.shuffle(seed=seed).select(range(subset_size))
    return [row["text"] for row in dataset if "text" in row]


def _unzip_eval_if_needed(test_root: Path) -> None:
    
    zip_path = test_root / "NLP26_OWT_eval.zip"
    if not zip_path.exists():
        return

    target_dir = test_root / "NLP26_OWT_eval"
    if target_dir.exists():
        return

    with zipfile.ZipFile(zip_path, "r") as handle:
        handle.extractall(test_root)


def load_test_split_sentences(test_root: Path) -> set[str]:
    
    _unzip_eval_if_needed(test_root)

    candidates = list(test_root.rglob("*.txt"))
    if not candidates:
        raise FileNotFoundError(
            "No test split text files found under "
            f"{test_root.as_posix()}."
        )

    sentences: set[str] = set()
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="ignore")
        for sentence in split_into_sentences(text):
            normalized = normalize_text(sentence)
            if normalized:
                sentences.add(normalized)
    return sentences


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple punctuation heuristic.

    Returns:
        A list of sentence strings.
    """
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in pieces if p.strip()]


def normalize_text(text: str) -> str:
    """Normalize text for comparison by lowercasing and collapsing whitespace.

    This reduces false negatives when checking for overlaps between train and
    test sentences.
    """
    return re.sub(r"\s+", " ", text.lower()).strip()


def looks_english(text: str) -> bool:
    """Heuristic English filter using ASCII ratio and common-word hits.

    This is intentionally lightweight to avoid heavy dependencies. It will
    misclassify short or code-like snippets and some non-English text.
    """
    if not text:
        return False

    ascii_chars = sum(1 for ch in text if ord(ch) < 128)
    ascii_ratio = ascii_chars / max(len(text), 1)
    if ascii_ratio < 0.9:
        return False

    common_words = {"the", "and", "to", "of", "in", "is", "for", "that"}
    tokens = re.findall(r"[a-z]+", text.lower())
    if not tokens:
        return False

    hits = sum(1 for token in tokens if token in common_words)
    return hits >= 2


def clean_text(text: str) -> str:
    """Remove HTML, code snippets, and special characters.

    Cleaning prioritizes readability and compatibility with downstream tokenizers.
    It strips HTML tags, inline/backtick code, fenced code blocks, indented code,
    and non-alphanumeric symbols outside a conservative punctuation set.
    """
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^\s{4,}.*$", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s\.\,\!\?\;\:\'\"\-\(\)]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_overlap(
    texts: list[str],
    test_sentences: set[str],
) -> list[str]:
    """Drop any texts that contain sentences seen in the test split.

    Leakage prevention uses sentence-level matching on normalized text to catch
    overlaps even when whitespace or casing differs.
    """
    filtered: list[str] = []
    for text in texts:
        sentences = split_into_sentences(text)
        normalized_sentences = {
            normalize_text(sentence) for sentence in sentences if sentence.strip()
        }
        if normalized_sentences & test_sentences:
            continue
        filtered.append(text)
    return filtered


def preprocess_openwebtext(
    test_root: Path,
    output_path: Path,
    subset_size: int = 5000,
    seed: int = 42,
    streaming: bool = False,
) -> dict[str, int]:
   
    texts = load_owt_subset(
        name="Skylion007/openwebtext",
        split="train",
        subset_size=subset_size,
        seed=seed,
        streaming=streaming,
    )
    test_sentences = load_test_split_sentences(test_root)
    no_overlap = remove_overlap(texts, test_sentences)

    cleaned: list[str] = []
    for text in no_overlap:
        cleaned_text = clean_text(text)
        if not cleaned_text:
            continue
        # English filtering happens after cleaning to avoid HTML/code artifacts.
        if not looks_english(cleaned_text):
            continue
        cleaned.append(cleaned_text)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in cleaned:
            handle.write(json.dumps({"text": item}, ensure_ascii=False))
            handle.write("\n")

    return {
        "subset_loaded": len(texts),
        "test_sentences": len(test_sentences),
        "after_overlap": len(no_overlap),
        "after_cleaning": len(cleaned),
    }

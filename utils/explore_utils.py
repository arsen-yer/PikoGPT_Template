"""
Lightweight helpers to explore text datasets for GPT-style pretraining.

The functions are designed to be notebook-friendly: minimal arguments,
safe fallbacks when optional libraries are missing, and small dict outputs
you can pretty-print or log.
"""

from __future__ import annotations

import os
import re
from collections import Counter
from typing import Iterable, List, Tuple

from datasets import load_dataset


# ---------- Loading ----------

def load_samples(
    dataset_name: str = "NLP26_OpenWebText",
    fallback_name: str = "openwebtext",
    split: str = "train",
    sample_size: int = 2000,
    cache_dir: str = "dataset/hf_cache",
    streaming: bool = False,
) -> Tuple[List[str], str]:
    """
    Load up to `sample_size` text rows from a dataset, with a fallback name.
    Returns (samples, resolved_dataset_name).
    """
    os.makedirs(cache_dir, exist_ok=True)

    def _load(name: str):
        return load_dataset(name, split=split, cache_dir=cache_dir, streaming=streaming)

    try:
        ds = _load(dataset_name)
        resolved = dataset_name
    except Exception:
        ds = _load(fallback_name)
        resolved = fallback_name

    if streaming:
        samples: List[str] = []
        for i, ex in enumerate(ds):
            samples.append(ex.get("text", ""))
            if i + 1 >= sample_size:
                break
    else:
        n = sample_size if sample_size is not None else len(ds)
        n = min(n, len(ds))
        ds_subset = ds.select(range(n))
        samples = ds_subset["text"]

    return samples, resolved


# ---------- Stats helpers ----------

def length_stats(texts: Iterable[str]) -> dict:
    lengths = [len(t) for t in texts]
    if not lengths:
        return {"count": 0}
    avg = sum(lengths) / len(lengths)
    short = sum(1 for l in lengths if l < 200)
    return {
        "count": len(lengths),
        "min": min(lengths),
        "max": max(lengths),
        "avg": avg,
        "short_lt_200": short,
        "short_pct": 100 * short / len(lengths),
    }


def _get_tokenizer(name: str = "gpt2"):
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # pragma: no cover - optional path
        return exc


def token_stats(texts: List[str], tokenizer_name: str = "gpt2", max_samples: int = 2000) -> dict:
    tok = _get_tokenizer(tokenizer_name)
    if isinstance(tok, Exception):
        return {"error": str(tok)}

    subset = texts[:max_samples]
    token_lens = [len(tok.encode(t, add_special_tokens=False)) for t in subset]
    if not token_lens:
        return {"count": 0}
    over_2048 = sum(1 for l in token_lens if l > 2048)
    return {
        "count": len(token_lens),
        "min": min(token_lens),
        "max": max(token_lens),
        "avg": sum(token_lens) / len(token_lens),
        "over_2048": over_2048,
        "over_2048_pct": 100 * over_2048 / len(token_lens),
    }


# ---------- Heuristics ----------

HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://|www\\.")
CTRL_RE = re.compile(r"[\\u0000-\\u001F\\u007F]")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\\+?\\d[\\d\\s().-]{8,}")
CODE_MARKERS = ["```", "{", "}", ";", "def ", "class ", "#include", "import "]


def looks_like_code(text: str) -> bool:
    if "```" in text:
        return True
    brace_count = text.count("{") + text.count("}")
    if brace_count >= 10:
        return True
    return sum(1 for m in CODE_MARKERS if m in text) >= 3


def non_english(text: str) -> bool:
    if not text:
        return False
    try:
        from langdetect import detect

        return detect(text) != "en"
    except Exception:
        non_ascii = sum(1 for ch in text if ord(ch) > 127)
        return (non_ascii / max(len(text), 1)) > 0.2


def flag_counts(texts: Iterable[str]) -> dict:
    counts = Counter()
    for t in texts:
        counts["html"] += bool(HTML_RE.search(t))
        counts["code"] += looks_like_code(t)
        counts["non_en"] += non_english(t)
        counts["url"] += bool(URL_RE.search(t))
        counts["ctrl"] += bool(CTRL_RE.search(t))
        counts["email"] += bool(EMAIL_RE.search(t))
        counts["phone"] += bool(PHONE_RE.search(t))
    total = len(list(texts)) if isinstance(texts, list) else None
    if total:
        counts["total"] = total
    return dict(counts)


# ---------- Duplicates ----------

def normalize(text: str) -> str:
    return re.sub(r"\\s+", " ", text.strip().lower())


def duplicate_stats(texts: List[str]) -> dict:
    normed = [normalize(t) for t in texts]
    total = len(normed)
    unique = len(set(normed))
    dupes = total - unique
    return {
        "total": total,
        "unique": unique,
        "dupes": dupes,
        "dupe_pct": 100 * dupes / total if total else 0.0,
    }


# ---------- Convenience wrapper ----------

def full_eda_report(samples: List[str]) -> dict:
    """Compute a bundle of EDA metrics on `samples`."""
    return {
        "lengths": length_stats(samples),
        "tokens": token_stats(samples),
        "flags": flag_counts(samples),
        "duplicates": duplicate_stats(samples),
    }


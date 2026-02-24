from __future__ import annotations

import html
import re


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple punctuation heuristic."""
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def normalize_text(text: str) -> str:
    """Normalize text for comparison by lowercasing and collapsing whitespace."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def looks_english(text: str) -> bool:
    """Heuristic English filter using ASCII ratio and common-word hits."""
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
    """Remove HTML, code snippets, and special characters."""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^\s{4,}.*$", " ", text)
    text = re.sub(r"[^A-Za-z0-9\s\.\,\!\?\;\:\'\"\-\(\)]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

"""
Notebook-friendly preprocessing helpers for GPT-style data cleaning.

All heavy dependencies are optional and imported lazily so the notebook
doesn't fail if a package is missing. Functions return simple Python
objects to keep usage straightforward in cells.
"""

from __future__ import annotations

import html
import re
from typing import Iterable, List, Optional, Sequence

# ---------- Regex primitives ----------

CTRL_RE = re.compile(r"[\x00-\x1F\x7F]")
HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
CODE_MARKERS = ("```", "{", "}", ";", "def ", "class ", "#include", "import ")


# ---------- Small helpers ----------

def remove_control_chars(text: str) -> str:
    """Strip control characters and collapse whitespace."""
    text = CTRL_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_html(text: str) -> str:
    """Remove HTML tags and unescape entities."""
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def looks_like_code(text: str) -> bool:
    """Heuristic to flag code-heavy samples."""
    if "```" in text:
        return True
    brace_count = text.count("{") + text.count("}")
    if brace_count >= 10:
        return True
    return sum(1 for m in CODE_MARKERS if m in text) >= 3


def replace_urls(
    text: str,
    placeholder: str = "<url>",
    url_placeholder: str | None = None,
) -> str:
    """Replace URLs with a placeholder. `url_placeholder` kept for backwards compatibility."""
    if url_placeholder is not None:
        placeholder = url_placeholder
    return URL_RE.sub(placeholder, text)


def is_english(text: str) -> bool:
    """Language check with langdetect fallback; otherwise heuristic by ASCII ratio."""
    if not text:
        return False
    try:
        from langdetect import detect
    except Exception:
        non_ascii = sum(1 for ch in text if ord(ch) > 127)
        return (non_ascii / max(len(text), 1)) <= 0.2
    try:
        return detect(text) == "en"
    except Exception:
        non_ascii = sum(1 for ch in text if ord(ch) > 127)
        return (non_ascii / max(len(text), 1)) <= 0.2


def load_tokenizer(name: str = "gpt2"):
    """Lazy-load a tokenizer; returns the tokenizer or an exception."""
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # pragma: no cover - optional path
        return exc


def truncate_tokens(text: str, tokenizer, max_tokens: int) -> str:
    """Truncate a string to max_tokens using the provided tokenizer."""
    if isinstance(tokenizer, Exception):
        return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids)


# ---------- Core preprocess ----------

def preprocess_text(
    text: str,
    *,
    tokenizer=None,
    max_tokens: Optional[int] = None,
    drop_code: bool = True,
    drop_non_english: bool = True,
    url_placeholder: str = "<url>",
    min_chars: int = 50,
    max_chars: int = 100_000,
) -> Optional[str]:
    """Apply a sequence of cleaning steps; return None if filtered out."""
    if not text:
        return None

    if drop_code and looks_like_code(text):
        return None

    text = strip_html(text)
    text = replace_urls(text, url_placeholder=url_placeholder)
    text = remove_control_chars(text)

    if drop_non_english and not is_english(text):
        return None

    if len(text) < min_chars or len(text) > max_chars:
        return None

    if max_tokens is not None and tokenizer is not None:
        text = truncate_tokens(text, tokenizer, max_tokens)

    return text if text else None


def preprocess_batch(
    texts: Sequence[str],
    *,
    tokenizer=None,
    max_tokens: Optional[int] = None,
    drop_code: bool = True,
    drop_non_english: bool = True,
    url_placeholder: str = "<url>",
    min_chars: int = 50,
    max_chars: int = 100_000,
) -> List[str]:
    """Preprocess a batch and drop filtered items."""
    cleaned: List[str] = []
    for t in texts:
        out = preprocess_text(
            t,
            tokenizer=tokenizer,
            max_tokens=max_tokens,
            drop_code=drop_code,
            drop_non_english=drop_non_english,
            url_placeholder=url_placeholder,
            min_chars=min_chars,
            max_chars=max_chars,
        )
        if out:
            cleaned.append(out)
    return cleaned


# ---------- Overlap filtering ----------

def load_test_sentences(test_paths: Iterable[str]) -> set:
    """Load test sentences (one string per line) from given paths."""
    sent_set: set = set()
    for path in test_paths:
        try:
            data = open(path, "r", encoding="utf-8", errors="ignore").read()
        except FileNotFoundError:
            continue
        for sent in re.split(r"(?<=[.!?])\s+", data.strip()):
            norm = re.sub(r"\s+", " ", sent.lower()).strip()
            if norm:
                sent_set.add(norm)
    return sent_set


def remove_overlap(texts: Sequence[str], test_sents: set) -> List[str]:
    """Drop any text containing a sentence present in test_sents."""
    keep: List[str] = []
    for t in texts:
        sentences = re.split(r"(?<=[.!?])\s+", t.strip())
        norm_sents = {re.sub(r"\s+", " ", s.lower()).strip() for s in sentences if s.strip()}
        if norm_sents & test_sents:
            continue
        keep.append(t)
    return keep

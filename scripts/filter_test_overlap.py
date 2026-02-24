from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

from datasets import load_dataset
from tqdm import tqdm

DEFAULT_TEST_URL = "https://drive.switch.ch/index.php/s/6TLGQFEIkAPJ72K"
DEFAULT_OUTPUT_PATH = Path("data/filtered/openwebtext_filtered.jsonl")
DEFAULT_REPORT_PATH = Path("data/filtered/filter_report.json")

SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.?!])\s+")
WHITESPACE_REGEX = re.compile(r"\s+")


def normalize_sentence(text: str) -> str:
    return WHITESPACE_REGEX.sub(" ", text.strip().lower())


def regex_split(text: str) -> list[str]:
    return [s for s in SENTENCE_SPLIT_REGEX.split(text) if s]


def nltk_split(text: str) -> list[str]:
    try:
        import nltk
    except ImportError as exc:
        raise RuntimeError("nltk is not installed. Install it or use --splitter regex.") from exc

    try:
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(text)
    except LookupError:
        nltk.download("punkt")
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(text)


def get_splitter(name: str):
    if name == "regex":
        return regex_split
    if name == "nltk":
        return nltk_split
    raise ValueError(f"Unknown splitter: {name}")


def download_test_file(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url) as response:
            content_type = response.headers.get("Content-Type", "")
            data = response.read()
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Failed to download test file: {exc}") from exc

    if b"<html" in data[:1000].lower() or "text/html" in content_type:
        raise RuntimeError(
            "Test URL returned HTML instead of a raw file. Download manually and pass --test_file."
        )

    dest.write_bytes(data)
    return dest


def load_test_sentences(
    test_path: Path,
    splitter_name: str,
    min_chars: int,
) -> set[str]:
    splitter = get_splitter(splitter_name)
    text = test_path.read_text(encoding="utf-8", errors="ignore")
    sentences = splitter(text)
    hashes: set[str] = set()
    for sentence in sentences:
        normalized = normalize_sentence(sentence)
        if len(normalized) < min_chars:
            continue
        digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
        hashes.add(digest)
    return hashes


def iter_openwebtext(
    subset_size: int,
    streaming: bool,
) -> Iterable[str]:
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=streaming)
    if streaming:
        count = 0
        for sample in dataset:
            if "text" not in sample:
                continue
            yield sample["text"]
            count += 1
            if count >= subset_size:
                break
        return

    dataset = dataset.select(range(subset_size))
    for text in dataset["text"]:
        yield text


def filter_openwebtext(
    test_hashes: set[str],
    subset_size: int,
    splitter_name: str,
    min_chars: int,
    streaming: bool,
    output_path: Path,
) -> dict[str, float | int]:
    splitter = get_splitter(splitter_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_rows_seen = 0
    total_rows_kept = 0
    total_sentences_seen = 0
    total_sentences_removed = 0

    start_time = time.time()
    with output_path.open("w", encoding="utf-8") as handle:
        for text in tqdm(iter_openwebtext(subset_size, streaming), total=subset_size):
            total_rows_seen += 1
            sentences = splitter(text)
            total_sentences_seen += len(sentences)
            kept_sentences = []
            for sentence in sentences:
                normalized = normalize_sentence(sentence)
                if len(normalized) < min_chars:
                    kept_sentences.append(sentence)
                    continue
                digest = hashlib.sha1(normalized.encode("utf-8")).hexdigest()
                if digest in test_hashes:
                    total_sentences_removed += 1
                    continue
                kept_sentences.append(sentence)

            filtered_text = " ".join(s.strip() for s in kept_sentences if s.strip())
            if not filtered_text:
                continue
            total_rows_kept += 1
            json.dump({"text": filtered_text}, handle, ensure_ascii=False)
            handle.write("\n")

    runtime_seconds = time.time() - start_time
    percent_sentences_removed = (
        (total_sentences_removed / total_sentences_seen) * 100
        if total_sentences_seen
        else 0.0
    )
    rows_dropped = total_rows_seen - total_rows_kept
    percent_rows_dropped = (rows_dropped / total_rows_seen) * 100 if total_rows_seen else 0.0

    return {
        "total_rows_seen": total_rows_seen,
        "total_rows_kept": total_rows_kept,
        "total_sentences_seen": total_sentences_seen,
        "total_sentences_removed": total_sentences_removed,
        "percent_sentences_removed": round(percent_sentences_removed, 4),
        "percent_rows_dropped": round(percent_rows_dropped, 4),
        "runtime_seconds": round(runtime_seconds, 2),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter OpenWebText by removing sentences that overlap with the test set."
    )
    parser.add_argument("--subset_size", type=int, default=5000, help="Rows to process")
    parser.add_argument(
        "--streaming",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Stream OpenWebText (recommended)",
    )
    parser.add_argument(
        "--test_url",
        type=str,
        default=DEFAULT_TEST_URL,
        help="URL to the Switch test set",
    )
    parser.add_argument(
        "--test_file",
        type=Path,
        default=None,
        help="Local test file path if manual download is needed",
    )
    parser.add_argument(
        "--splitter",
        choices=["regex", "nltk"],
        default="regex",
        help="Sentence splitter to use",
    )
    parser.add_argument(
        "--min_chars",
        type=int,
        default=20,
        help="Minimum normalized sentence length to consider for hashing",
    )
    parser.add_argument(
        "--output_path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output JSONL path",
    )
    parser.add_argument(
        "--report_path",
        type=Path,
        default=DEFAULT_REPORT_PATH,
        help="Report JSON path",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    test_path = None
    if args.test_file is not None:
        if not args.test_file.exists():
            print(
                "Test file not found. Download the Switch test set and pass --test_file.\n"
                f"Expected path: {args.test_file}",
                file=sys.stderr,
            )
            return 2
        test_path = args.test_file
    else:
        try:
            test_path = download_test_file(args.test_url, Path("data/raw/switch_test.txt"))
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            print(
                "Download the file manually from the Switch URL and rerun with --test_file.\n"
                f"URL: {args.test_url}",
                file=sys.stderr,
            )
            return 2

    test_hashes = load_test_sentences(
        test_path=test_path,
        splitter_name=args.splitter,
        min_chars=args.min_chars,
    )

    report = filter_openwebtext(
        test_hashes=test_hashes,
        subset_size=args.subset_size,
        splitter_name=args.splitter,
        min_chars=args.min_chars,
        streaming=args.streaming,
        output_path=args.output_path,
    )

    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

from src.preprocessing import preprocess_openwebtext


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OpenWebText preprocessing CLI.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["preprocess"],
        help="Pipeline stage to run.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5000,
        help="Number of samples to load from the subset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output",
        default="data/processed/openwebtext_clean.jsonl",
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--streaming",
        type=_parse_bool,
        default=True,
        help="Use HuggingFace streaming mode (true/false).",
    )
    return parser


def _run_preprocess(args: argparse.Namespace) -> None:
    if not args.streaming:
        print(
            "Warning: streaming is disabled. This may download many dataset shards."
        )
    stats = preprocess_openwebtext(
        output_path=Path(args.output),
        num_samples=args.num_samples,
        seed=args.seed,
        streaming=args.streaming,
        metadata_args={
            "num_samples": args.num_samples,
            "seed": args.seed,
            "output": args.output,
            "streaming": args.streaming,
        },
    )
    print(
        "Preprocessing summary: "
        f"loaded={stats['subset_loaded']} "
        f"after_cleaning={stats['after_cleaning']} "
        f"output={Path(args.output).as_posix()}"
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.stage == "preprocess":
        _run_preprocess(args)
        return
    raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()

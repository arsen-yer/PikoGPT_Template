# PikoGPT_Challenge

Lightweight template for the NLP with LLMs course (2026). Current goal: clean OpenWebText‑style data, avoid test leakage, and prepare tokenized shards.

## Requirements
- Python 3.11+
- `datasets`, `tqdm`, `transformers`, `langdetect`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
python -m pip install -e .
python -c "import datasets, transformers; print('ok')"
```

## EDA + Preprocessing
- `notebooks/EDA.ipynb` — quick look at raw data.
- `notebooks/dataset-preparation.ipynb` — main cleaning pipeline; set `MAX_DOCS` if you want to cap runtime.

Key helpers live in `utils/preprocessing_utils.py` (cleaning) and `utils/explore_utils.py` (EDA stats).

### Outputs
- Cache: `dataset/` (HF cache, ignored by git).
- Cleaned JSONL: choose `OUT_PATH` in the prep notebook (add `dataset_final/` to ignore if needed).
- Optional: tokenize/pack to 2048‑token blocks with the tokenization cell in the prep notebook.

### Test leakage
Provide test files in `TEST_PATHS` inside the prep notebook to drop any overlapping sentences (e.g., Wikitext103 test, NLP26 eval).

## Pre-training

## License

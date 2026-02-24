# PikoGPT_Challenge

This is a template script for the PikoGPT-Challange in the NLP with LLMs course semester 2026.

## Requirements

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager

## Environment

Use a single `.venv` for this project.

```bash
# Create and activate the environment
python -m venv .venv
source .venv/bin/activate

# Verify the active interpreter
python -c "import sys; print(sys.executable)"

# Install dependencies into the active interpreter
python -m pip install -e .

# Sanity check
python -c "import datasets; import transformers; print('ok')"
```

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
```

## Stages

| Stage | Status | Description |
|-------|--------|-------------|
| `data-preprocessing` | Implemented | Preprocess OpenWebText |
| `your-stage` | ? | This should be your stage |

## Data Loading And Filtering

Filtering test overlap is mandatory to avoid data leakage from the Switch test set.
Download the test file from the Switch URL and pass it via `--test_file`.

```text
https://drive.switch.ch/index.php/s/6TLGQFEIkAPJ72K
```

```bash
# Load a small OpenWebText subset (streaming)
python load_openwebtext.py

# Filter with mandatory test-set overlap removal
python scripts/filter_test_overlap.py --subset_size 5000 --test_file data/raw/nlp26_test.txt
```


## License

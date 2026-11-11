# Iris Species Classification

Random-forest classification from four flower measurements.

## Status and provenance

Educational project. The original notebook places this work under a CodSoft internship directory. The maintained `train.py` entry point was added during a portfolio maintenance pass with AI assistance; it is not a claim about the original internship deliverable. Original dataset redistribution rights and any tutorial source still need confirmation.

## Setup and run

Use Python 3.12. From this repository directory:

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on macOS/Linux, then:

```sh
python -m pip install -r requirements.txt
python train.py
```

The dataset is included in this repository. Use `--data PATH` for another file with the same columns. The script prints evaluation results; no pretrained model or hosted API is supplied. `python train.py --help` lists options.

## Method and limitations

Preprocessing is fitted on training rows only. A fixed seed makes the split reproducible. Small educational benchmark; one holdout split does not establish generalisation to other datasets.

The original notebook is retained as a learning artefact. Cleared notebook outputs must be regenerated; use `train.py` for the maintained evaluation path.

## Verified run

Python 3.12, the pinned direct dependencies and included CSV: 30 held-out rows, accuracy 0.90 versus a most-frequent baseline of 0.3333 (seed 42). This is a small single-split demonstration, not a general performance guarantee.

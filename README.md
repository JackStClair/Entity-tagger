# Entity Tagger

![Build Status](https://img.shields.io/badge/build-not%20configured-lightgrey)
![Test Coverage](https://img.shields.io/badge/coverage-not%20tracked-lightgrey)
![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.5.1-ee4c2c)

BiLSTM-based named entity tagging pipeline for slot-tag prediction from utterances.

## Repository Contents

- `run.py`: Trains a BiLSTM tagger, evaluates validation F1, saves the best model, and writes predictions.
- `requirements.txt`: Python dependencies.
- `hw2_train.csv`: Training dataset.
- `hw2_test.csv`: Test dataset for prediction.
- `run.sh`: End-to-end shell script for zipped homework workflow (`hw2.zip` -> `record.txt` logs).

## Quick Start

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Run training and generate predictions

```bash
python run.py
```

This creates:

- `best_model.pt` (best checkpoint during training)
- `submission.csv` (predicted IOB slot tags for test utterances)

## Notes

- `NUM_EPOCHS` in `run.py` is currently set to `30` for faster runtime.
- For the original longer training setup, set `NUM_EPOCHS = 150`.
- The script reads `hw2_train.csv` and `hw2_test.csv` from the repository root.

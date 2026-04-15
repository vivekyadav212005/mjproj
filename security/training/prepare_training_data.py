# security/training/prepare_training_data.py
"""
Prepares the Phase 1A dataset splits for DeBERTa fine-tuning.

Reads the frozen train/val/test CSVs produced in Phase 1A and
converts them to HuggingFace Dataset format ready for the Trainer.

Verifies:
  - No overlap between train, val, and test splits
  - Class distribution is documented
  - Output is saved to security/training/hf_dataset/

Usage:
    python security/training/prepare_training_data.py
"""

import os
import pandas as pd
from datasets import Dataset, DatasetDict

TRAIN_PATH = "data/processed/final_split/train.csv"
VAL_PATH   = "data/processed/final_split/val.csv"
TEST_PATH  = "data/processed/final_split/test.csv"
OUTPUT_DIR = "security/training/hf_dataset"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_and_verify() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Loading Phase 1A splits...")
    train = pd.read_csv(TRAIN_PATH)
    val   = pd.read_csv(VAL_PATH)
    test  = pd.read_csv(TEST_PATH)

    print(f"  Train : {len(train):,} rows")
    print(f"  Val   : {len(val):,} rows")
    print(f"  Test  : {len(test):,} rows")

    # Verify no leakage
    train_texts = set(train.text.str.lower().str.strip())
    val_texts   = set(val.text.str.lower().str.strip())
    test_texts  = set(test.text.str.lower().str.strip())

    tv = len(train_texts & val_texts)
    tt = len(train_texts & test_texts)
    vt = len(val_texts   & test_texts)
    assert tv == 0, f"Train/Val overlap: {tv} rows"
    assert tt == 0, f"Train/Test overlap: {tt} rows"
    assert vt == 0, f"Val/Test overlap: {vt} rows"
    print("  No data leakage between splits. ✓")

    # Class distribution
    for name, df in [("Train", train), ("Val", val), ("Test", test)]:
        b   = (df.label == 0).sum()
        inj = (df.label == 1).sum()
        print(f"  {name}: benign={b:,} ({b/len(df)*100:.1f}%) "
              f"injection={inj:,} ({inj/len(df)*100:.1f}%)")

    return train, val, test


def to_hf_dataset(df: pd.DataFrame) -> Dataset:
    """Convert a DataFrame to HuggingFace Dataset with text and label columns."""
    return Dataset.from_dict({
        "text":  df["text"].tolist(),
        "label": df["label"].astype(int).tolist(),
    })


def main():
    print("=" * 55)
    print("Preparing DeBERTa Training Data from Phase 1A Splits")
    print("=" * 55)

    train_df, val_df, test_df = load_and_verify()

    dataset = DatasetDict({
        "train":      to_hf_dataset(train_df),
        "validation": to_hf_dataset(val_df),
        "test":       to_hf_dataset(test_df),
    })

    dataset.save_to_disk(OUTPUT_DIR)
    print(f"\nDataset saved to {OUTPUT_DIR}")
    print(f"  Train : {len(dataset['train']):,}")
    print(f"  Val   : {len(dataset['validation']):,}")
    print(f"  Test  : {len(dataset['test']):,}")
    print("\nReady for train_deberta.py")


if __name__ == "__main__":
    main()
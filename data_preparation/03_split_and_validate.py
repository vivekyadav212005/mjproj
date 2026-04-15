# data_preparation/03_split_and_validate.py
"""
Phase 1A — Step 3: Create the frozen train/val/test split.

Split: 70% train / 15% validation / 15% test
Stratified: maintains class ratio in every split.
Frozen: test set is saved separately and NEVER used during model development.

This split is the foundation of reproducibility. The exact random seed (42)
must be documented in the paper.

Run: python data_preparation/03_split_and_validate.py
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH  = "data/processed/merged_deduped.csv"
TRAIN_PATH  = "data/processed/final_split/train.csv"
VAL_PATH    = "data/processed/final_split/val.csv"
TEST_PATH   = "data/processed/final_split/test.csv"
RANDOM_SEED = 42

os.makedirs("data/processed/final_split", exist_ok=True)


def main():
    print(f"Loading {INPUT_PATH}...")
    df = pd.read_csv(INPUT_PATH)
    print(f"Total samples: {len(df):,}")

    # Step 1: hold out 15% as test (frozen)
    train_val, test = train_test_split(
        df,
        test_size=0.15,
        stratify=df["label"],
        random_state=RANDOM_SEED
    )

    # Step 2: split remaining into train (70% of total) and val (15% of total)
    # 15% of total = 15/85 ≈ 17.6% of train_val
    val_fraction = 0.15 / 0.85
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=train_val["label"],
        random_state=RANDOM_SEED
    )

    # Save
    train.to_csv(TRAIN_PATH, index=False)
    val.to_csv(VAL_PATH,     index=False)
    test.to_csv(TEST_PATH,   index=False)

    print("\n" + "="*60)
    print("SPLIT SUMMARY")
    print("="*60)
    for name, split_df in [("Train", train), ("Val", val), ("Test", test)]:
        n = len(split_df)
        b = (split_df.label==0).sum()
        inj = (split_df.label==1).sum()
        print(f"{name:6s}: {n:,} rows | benign={b:,} ({b/n*100:.1f}%) | injection={inj:,} ({inj/n*100:.1f}%)")

    print(f"\nTest split is FROZEN. Path: {TEST_PATH}")
    print(f"Random seed used: {RANDOM_SEED} (document in paper)")


if __name__ == "__main__":
    main()
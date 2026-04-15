# security/training/train_deberta.py
"""
Fine-tunes microsoft/deberta-v3-base on the Phase 1A prompt injection
dataset for binary classification (benign=0, injection=1).

This script produces the fine-tuned model used by S2 and S3.
Running this script is REQUIRED before running S2 or S3 evaluations.

Training configuration (documented in Section 3.4 of the paper):
  Base model      : microsoft/deberta-v3-base
  Task            : Binary sequence classification
  Max seq length  : 512 subword tokens
  Batch size      : 16 train / 32 eval
  Learning rate   : 2e-5 with linear warmup over 10% of steps
  Weight decay    : 0.01
  Epochs          : 3 with early stopping on validation F1
  Optimizer       : AdamW
  Loss            : Cross-entropy with class weights (n_benign/n_injection)
                    to correct for 58.5/41.5 class imbalance
  Evaluation      : 5-fold cross-validation on training data
                    + final evaluation on frozen test split

Outputs:
  security/training/fine_tuned_model/   ← trained model weights
  security/training/training_report.txt ← metrics for Table 1 in paper

Hardware note:
  CPU  : ~8-12 hours (not recommended for full run)
  MPS  : ~2-4 hours  (Apple Silicon MacBook)
  CUDA : ~30-60 min  (Google Colab free T4 GPU — RECOMMENDED)

Google Colab instructions (free, recommended):
  1. Upload your mjproj/ directory to Google Drive
  2. Open a new Colab notebook
  3. Mount Drive: from google.colab import drive; drive.mount('/content/drive')
  4. cd /content/drive/MyDrive/mjproj
  5. pip install transformers torch datasets scikit-learn
  6. python security/training/train_deberta.py
  7. Download security/training/fine_tuned_model/ back to your machine

Usage:
    python security/training/prepare_training_data.py   # run first
    python security/training/train_deberta.py
"""

import os
import json
import numpy as np
import torch
from datasets import load_from_disk, DatasetDict, concatenate_datasets
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths and hyperparameters
# ─────────────────────────────────────────────────────────────────────────────

DATASET_DIR    = "security/training/hf_dataset"
MODEL_NAME     = "microsoft/deberta-v3-base"
OUTPUT_DIR     = "security/training/fine_tuned_model"
REPORT_PATH    = "security/training/training_report.txt"
CHECKPOINT_DIR = "security/training/checkpoints"

MAX_LENGTH     = 512
TRAIN_BATCH    = 16
EVAL_BATCH     = 32
LEARNING_RATE  = 2e-5
WEIGHT_DECAY   = 0.01
NUM_EPOCHS     = 3
WARMUP_RATIO   = 0.10
N_FOLDS        = 5
RANDOM_SEED    = 42

os.makedirs(OUTPUT_DIR,     exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────────────────────────────────────

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# Tokenisation
# ─────────────────────────────────────────────────────────────────────────────

def tokenise_dataset(dataset: DatasetDict, tokenizer) -> DatasetDict:
    def tokenise(batch):
        return tokenizer(
            batch["text"],
            truncation   = True,
            max_length   = MAX_LENGTH,
            padding      = "max_length",
        )
    return dataset.map(tokenise, batched=True,
                       remove_columns=["text"],
                       desc="Tokenising")


# ─────────────────────────────────────────────────────────────────────────────
# Class weights for imbalanced dataset
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(labels: list[int]) -> torch.Tensor:
    """
    Compute class weights = n_benign / n_injection for injection class.
    This corrects for the 58.5/41.5 class imbalance in the training data.
    """
    n_total    = len(labels)
    n_benign   = sum(1 for l in labels if l == 0)
    n_injection = sum(1 for l in labels if l == 1)
    weight_benign   = n_total / (2 * n_benign)
    weight_injection = n_total / (2 * n_injection)
    print(f"  Class weights: benign={weight_benign:.4f} "
          f"injection={weight_injection:.4f}")
    return torch.tensor([weight_benign, weight_injection])


# ─────────────────────────────────────────────────────────────────────────────
# Custom Trainer with weighted loss
# ─────────────────────────────────────────────────────────────────────────────

class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = self.args.device if hasattr(self.args, 'device') else "cpu"
        self.class_weights = class_weights.to(device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits
        weights = self.class_weights.to(dtype=logits.dtype, device=logits.device)
        loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
        loss    = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ─────────────────────────────────────────────────────────────────────────────
# Metrics computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(eval_pred) -> dict:
    logits, labels  = eval_pred
    # Replace any NaN/Inf in logits before softmax (guards against exploding gradients)
    logits          = np.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
    probs           = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    preds           = np.argmax(logits, axis=-1)
    injection_probs = probs[:, 1]

    # roc_auc_score requires valid probabilities — fall back to 0.0 if still broken
    try:
        roc_auc = roc_auc_score(labels, injection_probs)
    except ValueError:
        roc_auc = 0.0

    return {
        "accuracy":  accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall":    recall_score(labels, preds, zero_division=0),
        "f1":        f1_score(labels, preds, zero_division=0),
        "roc_auc":   roc_auc,
        "fnr":       1 - recall_score(labels, preds, zero_division=0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5-Fold Cross Validation
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_validation(train_dataset, tokenizer, class_weights) -> dict:
    """
    5-fold cross-validation on the training data.
    Reports mean ± std for all metrics (required for journal submission).
    """
    print(f"\n{'='*55}")
    print(f"5-Fold Cross-Validation on Training Data")
    print(f"{'='*55}")

    labels   = train_dataset["label"]
    indices  = list(range(len(train_dataset)))
    skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                random_state=RANDOM_SEED)

    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(
            skf.split(indices, labels), 1):
        print(f"\n  Fold {fold}/{N_FOLDS}")

        fold_train = train_dataset.select(train_idx)
        fold_val   = train_dataset.select(val_idx)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=2,
            ignore_mismatched_sizes=True,
        )

        n_steps      = len(fold_train) // TRAIN_BATCH * NUM_EPOCHS
        warmup_steps = int(n_steps * WARMUP_RATIO)

        args = TrainingArguments(
            output_dir              = os.path.join(CHECKPOINT_DIR,
                                                    f"fold_{fold}"),
            num_train_epochs        = NUM_EPOCHS,
            per_device_train_batch_size = TRAIN_BATCH,
            per_device_eval_batch_size  = EVAL_BATCH,
            learning_rate           = LEARNING_RATE,
            weight_decay            = WEIGHT_DECAY,
            warmup_steps            = warmup_steps,
            eval_strategy           = "epoch",
            save_strategy           = "epoch",
            load_best_model_at_end  = True,
            metric_for_best_model   = "f1",
            greater_is_better       = True,
            logging_steps           = 50,
            seed                    = RANDOM_SEED,
            report_to               = "none",
            fp16                    = False,
            max_grad_norm           = 1.0,
            dataloader_pin_memory   = False,
        )

        trainer = WeightedTrainer(
            class_weights = class_weights,
            model         = model,
            args          = args,
            train_dataset = fold_train,
            eval_dataset  = fold_val,
            compute_metrics = compute_metrics,
            callbacks     = [EarlyStoppingCallback(early_stopping_patience=1)],
        )
        trainer.train()

        metrics = trainer.evaluate()
        fold_metrics.append({
            "accuracy":  metrics.get("eval_accuracy", 0),
            "precision": metrics.get("eval_precision", 0),
            "recall":    metrics.get("eval_recall", 0),
            "f1":        metrics.get("eval_f1", 0),
            "roc_auc":   metrics.get("eval_roc_auc", 0),
            "fnr":       metrics.get("eval_fnr", 0),
        })
        print(f"  Fold {fold} F1: {fold_metrics[-1]['f1']:.4f}")

    # Aggregate
    summary = {}
    for metric in ["accuracy", "precision", "recall", "f1", "roc_auc", "fnr"]:
        vals = [m[metric] for m in fold_metrics]
        summary[metric] = {
            "mean": round(float(np.mean(vals)), 4),
            "std":  round(float(np.std(vals)),  4),
        }

    print(f"\n  CV Results (mean ± std):")
    for metric, stats in summary.items():
        print(f"    {metric:<12}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Final training on full train set + test evaluation
# ─────────────────────────────────────────────────────────────────────────────

def train_final_model(train_dataset, val_dataset, test_dataset,
                      tokenizer, class_weights) -> dict:
    """
    Train the final model on the full training set.
    Evaluate on the frozen test split.
    Save model to OUTPUT_DIR.
    """
    print(f"\n{'='*55}")
    print("Final Training on Full Training Set")
    print(f"{'='*55}")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2,
        ignore_mismatched_sizes=True,
    )

    n_steps      = len(train_dataset) // TRAIN_BATCH * NUM_EPOCHS
    warmup_steps = int(n_steps * WARMUP_RATIO)

    args = TrainingArguments(
        output_dir              = CHECKPOINT_DIR,
        num_train_epochs        = NUM_EPOCHS,
        per_device_train_batch_size = TRAIN_BATCH,
        per_device_eval_batch_size  = EVAL_BATCH,
        learning_rate           = LEARNING_RATE,
        weight_decay            = WEIGHT_DECAY,
        warmup_steps            = warmup_steps,
        eval_strategy           = "epoch",
        save_strategy           = "epoch",
        load_best_model_at_end  = True,
        metric_for_best_model   = "f1",
        greater_is_better       = True,
        logging_steps           = 100,
        seed                    = RANDOM_SEED,
        report_to               = "none",
        fp16                    = False,
        max_grad_norm           = 1.0,
        dataloader_pin_memory   = False,
    )

    trainer = WeightedTrainer(
        class_weights   = class_weights,
        model           = model,
        args            = args,
        train_dataset   = train_dataset,
        eval_dataset    = val_dataset,
        compute_metrics = compute_metrics,
        callbacks       = [EarlyStoppingCallback(early_stopping_patience=1)],
    )

    print("  Training...")
    trainer.train()

    # Evaluate on frozen test split (NEVER touched during development)
    print("\n  Evaluating on frozen test split...")
    test_metrics = trainer.evaluate(test_dataset)
    print(f"  Test F1       : {test_metrics.get('eval_f1', 0):.4f}")
    print(f"  Test ROC-AUC  : {test_metrics.get('eval_roc_auc', 0):.4f}")
    print(f"  Test FNR      : {test_metrics.get('eval_fnr', 0):.4f}")

    # Save model
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\n  Model saved to {OUTPUT_DIR}")

    return {
        "accuracy":  test_metrics.get("eval_accuracy", 0),
        "precision": test_metrics.get("eval_precision", 0),
        "recall":    test_metrics.get("eval_recall", 0),
        "f1":        test_metrics.get("eval_f1", 0),
        "roc_auc":   test_metrics.get("eval_roc_auc", 0),
        "fnr":       test_metrics.get("eval_fnr", 0),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────────────────────────

def write_report(cv_results: dict, test_results: dict):
    lines = [
        "=" * 60,
        "SECURERAG — ML CLASSIFIER TRAINING REPORT",
        "Table 1: Classifier Performance (for paper)",
        "=" * 60,
        "",
        f"Base model : {MODEL_NAME}",
        f"Dataset    : Phase 1A merged corpus (17,864 samples)",
        f"Split      : 70/15/15 train/val/test (seed=42, frozen test)",
        f"CV folds   : {N_FOLDS}",
        "",
        "5-FOLD CROSS-VALIDATION (on training data):",
        "-" * 60,
        f"  {'Metric':<14} {'Mean':>8} {'Std':>8}",
        "-" * 60,
    ]
    for metric, stats in cv_results.items():
        lines.append(
            f"  {metric:<14} {stats['mean']:>8.4f} {stats['std']:>8.4f}"
        )
    lines += [
        "",
        "FROZEN TEST SET EVALUATION:",
        "-" * 60,
        f"  {'Metric':<14} {'Value':>8}",
        "-" * 60,
    ]
    for metric, val in test_results.items():
        lines.append(f"  {metric:<14} {val:>8.4f}")
    lines += [
        "",
        "Note: FNR = False Negative Rate = 1 - Recall.",
        "  Lower FNR = fewer missed injections = better security.",
        "=" * 60,
    ]

    report = "\n".join(lines)
    with open(REPORT_PATH, "w") as f:
        f.write(report)
    print(f"\nReport saved to {REPORT_PATH}")
    print("\n" + report)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("Phase — DeBERTa Fine-Tuning for Prompt Injection Detection")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    if device == "cpu":
        print("\nWARNING: Running on CPU will take 8-12 hours.")
        print("Strongly recommended to run on Google Colab (free T4 GPU).")
        print("See script header for Colab instructions.")
        response = input("Continue on CPU? (yes/no): ")
        if response.lower() != "yes":
            print("Aborted. Upload to Colab and run there.")
            return

    # Load dataset
    print(f"\nLoading dataset from {DATASET_DIR}...")
    if not os.path.exists(DATASET_DIR):
        print(f"ERROR: {DATASET_DIR} not found.")
        print("Run: python security/training/prepare_training_data.py first")
        return

    dataset  = load_from_disk(DATASET_DIR)
    train_ds = dataset["train"]
    val_ds   = dataset["validation"]
    test_ds  = dataset["test"]
    print(f"  Train: {len(train_ds):,} | Val: {len(val_ds):,} "
          f"| Test: {len(test_ds):,}")

    # Load tokeniser
    print(f"\nLoading tokeniser: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenise
    print("Tokenising datasets...")
    full_dataset = DatasetDict({
        "train":      train_ds,
        "validation": val_ds,
        "test":       test_ds,
    })
    tokenised = tokenise_dataset(full_dataset, tokenizer)

    # Class weights
    print("\nComputing class weights...")
    labels         = tokenised["train"]["label"]
    class_weights  = compute_class_weights(labels)

    # 5-fold CV
    cv_results = run_cross_validation(
        tokenised["train"], tokenizer, class_weights
    )

    # Final model
    test_results = train_final_model(
        tokenised["train"],
        tokenised["validation"],
        tokenised["test"],
        tokenizer,
        class_weights,
    )

    # Save results JSON for ml_classifier.py to read
    results = {"cv": cv_results, "test": test_results}
    with open(os.path.join(OUTPUT_DIR, "training_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    write_report(cv_results, test_results)
    print(f"\nTraining complete. Fine-tuned model at: {OUTPUT_DIR}")
    print("You can now run S2 and S3 evaluations.")


if __name__ == "__main__":
    main()
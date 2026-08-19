#!/usr/bin/env python3
"""
Train TF-IDF + Logistic Regression for S2 and calibrate the threshold.

Threshold criterion (declared before seeing ASR, see
results/s2_audit/S2_training_pipeline_audit.md):

  Among candidate thresholds {0.10, 0.15, ..., 0.90}, select the value
  that maximises validation F1 subject to validation FPR <= 0.05.
  If none meet the FPR cap, take lowest FPR among recall >= 0.70.
  If still none, take minimum validation FPR.

Unconstrained max-F1 is reported but not used when it violates the cap.
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPLIT_DIR = os.path.join(REPO, "data", "processed", "s2_split")
MODEL_PATH = os.path.join(REPO, "security", "training", "s2_tfidf_lr.joblib")
METRICS_PATH = os.path.join(REPO, "results", "s2_audit", "s2_classifier_metrics.json")
THRESHOLD_PATH = os.path.join(REPO, "results", "s2_audit", "s2_threshold_selection.md")

CANDIDATE_THRESHOLDS = [round(x, 2) for x in np.arange(0.10, 0.91, 0.05)]
FPR_CAP = 0.05
RECALL_FLOOR = 0.70


def metrics_at_threshold(y_true, proba, thr):
    y_pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "fpr": float(fpr),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
    }


def select_threshold(rows):
    constrained = [r for r in rows if r["fpr"] <= FPR_CAP]
    if constrained:
        chosen = max(constrained, key=lambda r: (r["f1"], -r["fpr"], r["threshold"]))
        rule = (
            f"max validation F1 among thresholds with FPR <= {FPR_CAP:.2f}"
        )
        return chosen, rule
    recall_ok = [r for r in rows if r["recall"] >= RECALL_FLOOR]
    if recall_ok:
        chosen = min(recall_ok, key=lambda r: (r["fpr"], -r["f1"]))
        rule = (
            f"no threshold met FPR <= {FPR_CAP:.2f}; "
            f"lowest FPR among recall >= {RECALL_FLOOR:.2f}"
        )
        return chosen, rule
    chosen = min(rows, key=lambda r: (r["fpr"], -r["f1"]))
    rule = (
        f"no threshold met FPR <= {FPR_CAP:.2f} or recall >= {RECALL_FLOOR:.2f}; "
        "minimum validation FPR"
    )
    return chosen, rule


def main():
    train = pd.read_csv(os.path.join(SPLIT_DIR, "train.csv"))
    val = pd.read_csv(os.path.join(SPLIT_DIR, "val.csv"))
    test = pd.read_csv(os.path.join(SPLIT_DIR, "test.csv"))

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=50000,
        sublinear_tf=True,
        lowercase=True,
    )
    clf = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    X_train = vectorizer.fit_transform(train["text"].astype(str))
    clf.fit(X_train, train["label"].astype(int).to_numpy())

    def proba(df):
        return clf.predict_proba(vectorizer.transform(df["text"].astype(str)))[:, 1]

    val_p = proba(val)
    test_p = proba(test)
    y_val = val["label"].astype(int).to_numpy()
    y_test = test["label"].astype(int).to_numpy()

    val_rows = [metrics_at_threshold(y_val, val_p, t) for t in CANDIDATE_THRESHOLDS]
    unconstrained = max(val_rows, key=lambda r: r["f1"])
    chosen, rule = select_threshold(val_rows)
    test_at_chosen = metrics_at_threshold(y_test, test_p, chosen["threshold"])

    bundle = {
        "vectorizer": vectorizer,
        "model": clf,
        "threshold": chosen["threshold"],
        "selection_rule": rule,
        "tfidf": {
            "ngram_range": [1, 2],
            "min_df": 2,
            "max_df": 0.95,
            "max_features": 50000,
            "sublinear_tf": True,
        },
        "logreg": {
            "C": 1.0,
            "class_weight": "balanced",
            "max_iter": 1000,
            "solver": "lbfgs",
            "random_state": 42,
        },
    }
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)

    report = {
        "n_train": int(len(train)),
        "n_val": int(len(val)),
        "n_test": int(len(test)),
        "validation_sweep": val_rows,
        "unconstrained_max_f1_val": unconstrained,
        "selected_threshold": chosen,
        "selection_rule": rule,
        "fpr_cap": FPR_CAP,
        "test_at_selected_threshold": test_at_chosen,
        "model_path": MODEL_PATH,
    }
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(report, f, indent=2)

    lines = [
        "# S2 threshold selection",
        "",
        f"**Criterion (a priori):** {rule}.",
        f"**FPR cap:** {FPR_CAP:.2f}. **Recall floor (fallback):** {RECALL_FLOOR:.2f}.",
        f"**Selected threshold:** `{chosen['threshold']}`",
        "",
        "This threshold was not chosen to produce a target attack-success rate.",
        "",
        "| threshold | precision | recall | F1 | FPR |",
        "|----------:|----------:|-------:|---:|----:|",
    ]
    for r in val_rows:
        mark = " **← selected**" if r["threshold"] == chosen["threshold"] else ""
        lines.append(
            f"| {r['threshold']:.2f} | {r['precision']:.4f} | {r['recall']:.4f} "
            f"| {r['f1']:.4f} | {r['fpr']:.4f} |{mark}"
        )
    lines += [
        "",
        "## Unconstrained max-F1 (not used if it violates the FPR cap)",
        "",
        json.dumps(unconstrained, indent=2),
        "",
        "## Held-out test metrics at the selected threshold",
        "",
        json.dumps(test_at_chosen, indent=2),
        "",
        f"Model written to `{MODEL_PATH}`.",
    ]
    with open(THRESHOLD_PATH, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("Selected threshold:", chosen["threshold"])
    print("Rule:", rule)
    print("Val  P/R/F1/FPR:",
          f"{chosen['precision']:.3f}/{chosen['recall']:.3f}/"
          f"{chosen['f1']:.3f}/{chosen['fpr']:.3f}")
    print("Test P/R/F1/FPR:",
          f"{test_at_chosen['precision']:.3f}/{test_at_chosen['recall']:.3f}/"
          f"{test_at_chosen['f1']:.3f}/{test_at_chosen['fpr']:.3f}")
    print("Saved", MODEL_PATH)
    print("Saved", METRICS_PATH)
    print("Saved", THRESHOLD_PATH)


if __name__ == "__main__":
    main()

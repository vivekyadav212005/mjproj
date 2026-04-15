# security/ml_classifier.py
"""
S2 — ML-Based Prompt Injection Classifier (Fine-tuned DeBERTa-v3-base).

Loads the model fine-tuned by security/training/train_deberta.py on
the Phase 1A documented dataset. This is the correct approach for a
publishable paper because:
  - Training data is fully documented (Section 3.2 of the paper)
  - Test set was frozen before development began
  - We can prove our evaluation set is disjoint from training data
  - The fine-tuning is a reproducible contribution

If the fine-tuned model is not yet available (i.e. training has not
been run), the classifier raises a clear error rather than silently
falling back to a pre-trained checkpoint.

PREREQUISITE:
    python security/training/prepare_training_data.py
    python security/training/train_deberta.py

Then S2 and S3 evaluations can be run.
"""

import os
import torch
from transformers import pipeline

FINE_TUNED_MODEL_DIR = "security/training/fine_tuned_model"
THRESHOLD            = 0.5
MAX_LENGTH           = 512


class MLClassifier:
    """
    Fine-tuned DeBERTa-v3-base binary sequence classifier.

    Loads weights from the locally fine-tuned model directory.
    Raises FileNotFoundError if training has not been run yet.
    """

    def __init__(self,
                 model_dir: str = FINE_TUNED_MODEL_DIR,
                 threshold: float = THRESHOLD):
        self.threshold = threshold
        self.model_dir = model_dir

        if not os.path.exists(model_dir):
            raise FileNotFoundError(
                f"Fine-tuned model not found at '{model_dir}'.\n"
                "You must train the model before running S2/S3 evaluation.\n"
                "Run these scripts in order:\n"
                "  python security/training/prepare_training_data.py\n"
                "  python security/training/train_deberta.py\n"
                "(Recommended: run train_deberta.py on Google Colab free GPU)"
            )

        self._device   = self._select_device()
        self._pipeline = self._load_pipeline()

    @staticmethod
    def _select_device() -> str:
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"

    def _load_pipeline(self):
        print(f"  Loading fine-tuned ML classifier from {self.model_dir}")
        print(f"  Device: {self._device}")
        clf = pipeline(
            "text-classification",
            model     = self.model_dir,
            tokenizer = self.model_dir,
            device    = self._device,
            truncation= True,
            max_length= MAX_LENGTH,
        )
        print("  ML classifier loaded.")
        return clf

    def classify(self, text: str) -> dict:
        """
        Classify a single text string.

        Returns:
            score  : float — P(injection)
            label  : str   — "injection" or "benign"
            blocked: bool  — score >= threshold
        """
        if not text or not text.strip():
            return {"score": 0.0, "label": "benign", "blocked": False}

        result = self._pipeline(text[:4096])
        raw    = result[0]
        label  = raw["label"].lower()
        conf   = float(raw["score"])

        # Label 1 = injection in our fine-tuned model
        # The pipeline returns the predicted label and its confidence
        if "1" in label or "injection" in label or "label_1" in label:
            injection_score = conf
        else:
            injection_score = 1.0 - conf

        return {
            "score":   round(injection_score, 4),
            "label":   "injection" if injection_score >= self.threshold
                       else "benign",
            "blocked": injection_score >= self.threshold,
        }

    def classify_batch(self, texts: list[str]) -> list[dict]:
        return [self.classify(t) for t in texts]
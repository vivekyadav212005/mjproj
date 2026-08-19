import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "training", "s2_tfidf_lr.joblib"
)
LEGACY_THRESHOLD = 0.2


class SimpleMLDetector:
    """
    TF-IDF + logistic regression prompt-injection detector.

    Preferred path (S2): load a persisted bundle trained on query- and
    document-shaped injection data (see security/training/train_simple_ml.py).

    Legacy path (S3_fixed, unchanged): train() still fits on an S0
    attack-results JSONL of LLM *responses* and uses threshold 0.2.
    That path is kept so HybridDefendedRAG behaviour is not rewritten.
    """

    def __init__(self, model_path=None, threshold=None):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()
        self.threshold = LEGACY_THRESHOLD if threshold is None else float(threshold)
        self.trained = False
        self.selection_rule = None
        if model_path:
            self.load(model_path)

    def load(self, model_path: str):
        bundle = joblib.load(model_path)
        self.vectorizer = bundle["vectorizer"]
        self.model = bundle["model"]
        self.threshold = float(bundle["threshold"])
        self.selection_rule = bundle.get("selection_rule")
        self.trained = True

    def train(self, filepath):
        """Legacy S0-response trainer. Used by HybridDefendedRAG only."""
        texts = []
        labels = []

        with open(filepath, "r") as f:
            for line in f:
                data = json.loads(line)

                text = data.get("response", "")
                success = data.get("success", False)

                if text.strip():
                    texts.append(text)
                    labels.append(1 if success else 0)

        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.threshold = LEGACY_THRESHOLD
        self.trained = True
        print("Simple ML detector trained.")

    def classify(self, text: str) -> dict:
        if not self.trained or not text or not str(text).strip():
            return {"score": 0.0, "blocked": False}

        X = self.vectorizer.transform([text])
        prob = self.model.predict_proba(X)[0][1]

        return {
            "score": round(float(prob), 4),
            "blocked": bool(prob >= self.threshold),
        }

import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class SimpleMLDetector:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.model = LogisticRegression()
        self.trained = False

    def train(self, filepath):
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

        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.trained = True
        print("Simple ML detector trained.")

    def classify(self, text: str) -> dict:
        if not self.trained or not text.strip():
            return {"score": 0.0, "blocked": False}

        X = self.vectorizer.transform([text])
        prob = self.model.predict_proba(X)[0][1]

        return {
            "score": round(float(prob), 4),
            "blocked": prob >= 0.2
        }
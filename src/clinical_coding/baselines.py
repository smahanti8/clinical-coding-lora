from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def majority_class_predict(train_labels: list[str], n: int) -> list[str]:
    majority = Counter(train_labels).most_common(1)[0][0]
    return [majority] * n


@dataclass
class TfidfLogisticRegressionBaseline:
    vectorizer: TfidfVectorizer
    model: LogisticRegression

    @classmethod
    def fit(cls, texts: list[str], labels: list[str], seed: int = 0) -> "TfidfLogisticRegressionBaseline":
        vectorizer = TfidfVectorizer()
        features = vectorizer.fit_transform(texts)
        model = LogisticRegression(max_iter=1000, random_state=seed)
        model.fit(features, labels)
        return cls(vectorizer=vectorizer, model=model)

    def predict(self, texts: list[str]) -> list[str]:
        features = self.vectorizer.transform(texts)
        return list(self.model.predict(features))

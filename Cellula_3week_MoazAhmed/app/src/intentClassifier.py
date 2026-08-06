from transformers import pipeline

class IntentClassifier:
    def __init__(self):
        self.intentModel = pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")
        self.labels = ["generate code", "explain code"]

    def classify(self, query: str) -> str:
        res = self.intentModel(query, self.labels)
        return res['labels'][0]
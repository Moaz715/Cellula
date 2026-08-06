from sentence_transformers import CrossEncoder

class RelevanceGrader:
    def __init__(self, threshold: float = 2.0):
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.threshold = threshold

    def check_relevance(self, query: str, document_text: str) -> bool:
        try:
            score = self.model.predict([(query, document_text)])[0]
            print(f"[Grader] Score: {score:.2f} | Relevant: {score > self.threshold}")
            
            return score > self.threshold
            
        except Exception as e:
            print(f"[Error] Cross-Encoder Grader failed: {e}")
            return False
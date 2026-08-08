# src/grader.py
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_HOME'] = os.getenv("D_PATH")
from sentence_transformers import CrossEncoder

class RelevanceGrader:

    def __init__(self, threshold: float = 0.85):
        self.model = CrossEncoder('mixedbread-ai/mxbai-rerank-xsmall-v1')
        self.threshold = threshold

    def check_relevance(self, query: str, document_text: str) -> bool:
        try:
            score = float(self.model.predict([(query, document_text)])[0])
            is_relevant = score >= self.threshold
            
            
            
            return is_relevant
            
        except Exception as e:
            print(f"[Error] Cross-Encoder Grader failed: {e}")
            return False
# src/grader.py
import os
from dotenv import load_dotenv
load_dotenv()
os.environ['HF_HOME'] = os.getenv("D_PATH")
from sentence_transformers import CrossEncoder

class RelevanceGrader:
    # Set threshold to 0.0. BGE reranker outputs can be negative, 
    # but > 0.0 strongly indicates relevance for code.
    def __init__(self, threshold: float = 0.85):
        # Upgraded to the modern BAAI reranker!
        self.model = CrossEncoder('mixedbread-ai/mxbai-rerank-xsmall-v1')
        self.threshold = threshold

    def check_relevance(self, query: str, document_text: str) -> bool:
        try:
            score = float(self.model.predict([(query, document_text)])[0])
            is_relevant = score >= self.threshold
            
            preview = document_text.replace("\n", " ").strip()
            if len(preview) > 150:
                preview = preview[:150] + "..."

            print("=" * 60)
            print(f"🧐 [BGE RERANKER EVALUATION]")
            print(f"   Query : '{query}'")
            print(f"   Chunk : '{preview}'")
            print(f"   Score : {score:.4f} | Threshold: {self.threshold}")
            print(f"   Status: {'✅ PASSED' if is_relevant else '❌ REJECTED'}")
            print("=" * 60)
            
            return is_relevant
            
        except Exception as e:
            print(f"[Error] Cross-Encoder Grader failed: {e}")
            return False
import os
from dotenv import load_dotenv
load_dotenv()
d_path = os.getenv("D_PATH")
if d_path:
    os.environ['HF_HOME'] = d_path
from langchain_community.embeddings import HuggingFaceEmbeddings

class VectorEmbedder:
    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )

    def get_embedding_model(self):
        return self.model
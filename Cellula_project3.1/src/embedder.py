from langchain_community.embeddings import HuggingFaceEmbeddings

class Embedder:
    def __init__(self):
        self.model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"}
        )

    def get_embedding_model(self):
        return self.model
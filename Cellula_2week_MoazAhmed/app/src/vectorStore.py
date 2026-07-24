from langchain_classic.vectorstores import FAISS
from langchain_core.documents import Document

class VectorStoreManager:
    def __init__(self, chunks, embedding_model):
        self.vector_db = FAISS.from_documents(chunks, embedding_model)

    def similarity_search(self, query: str, k: int) -> list[str]:
        similar_docs = self.vector_db.similarity_search_with_score(query, k=k)
        context = [doc[0].page_content for doc in similar_docs]
        return context
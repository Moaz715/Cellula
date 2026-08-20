from langchain_chroma import Chroma
import uuid
class VectorStore:
    def __init__(self, embedder_model):
        self.db = Chroma(
            collection_name="rag_collection",
            embedding_function=embedder_model,
            persist_directory="./chroma_storage"
        )
        
    def add_chunks(self, chunked_docs):
        uuids = [str(uuid.uuid4()) for _ in range(len(chunked_docs))]
        self.db.add_documents(documents=chunked_docs, ids=uuids)
        
    def search(self, query: str, k: int = 4):
        docs = self.db.similarity_search(query, k=k)
        
        context_chunks = [doc.page_content for doc in docs]
        return context_chunks
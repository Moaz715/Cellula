from langchain_classic.vectorstores import Chroma
from langchain_core.documents import Document

class ChromaStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        
    def build(self, chunks, embedding_model):
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding_model=self.embedding_model,
            persist_directory='./chroma_data'
        )
        return self.vector_db
    
    def load(self):
        self.vector_db = Chroma(
            embedding_model=self.embedding_model,
            persist_directory='./chroma_data'
        )
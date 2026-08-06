from langchain_classic.vectorstores import Chroma
from langchain_core.documents import Document

class ChromaStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.vector_db = None
        
    def build(self, chunks):
        self.vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,  
            persist_directory='./chroma_data'
        )
        return self.vector_db
    
    def load(self):
        self.vector_db = Chroma(
            embedding_function=self.embedding_model,  
            persist_directory='./chroma_data'
        )
        return self.vector_db
    
    def add_document(self, code_text, metadata):
        if self.vector_db is None:
            self.load()
        doc = Document(page_content=code_text, metadata=metadata or {})
        self.vector_db.add_documents([doc])
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
class DocumentLoader:
    
    def load_data(self, path: str) -> list[Document]:
        loader = TextLoader(path)
        document = loader.load()
        return document
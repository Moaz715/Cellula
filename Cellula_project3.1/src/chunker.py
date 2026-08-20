from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document

class Chunker:
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language_map = {
            "py": Language.PYTHON,
            "js": Language.JS,
            "cpp": Language.CPP,
            "html": Language.HTML
        }
        
        
    def _get_type(self, doc: Document) -> str:
        
        if "type" in doc.metadata:
            return doc.metadata['type']
        
        src = doc.metadata.get('source', '')
        
        if src:
            return src.split('.')[-1].lower()
        
        return 'txt'
    
    
    def split_docs(self, docs: list[Document]) -> list[Document]:
        
        all_chunks = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        for doc in docs:
            file_type = self._get_type(doc)
            
            if file_type in self.language_map.keys():
                splitter = RecursiveCharacterTextSplitter.from_language(
                    language=self.language_map[file_type],
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
            else:
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size, 
                    chunk_overlap=self.chunk_overlap
                )
            
            chunks = splitter.split_documents([doc])
            all_chunks.extend(chunks)
            
        return all_chunks
            
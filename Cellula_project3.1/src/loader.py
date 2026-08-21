import os
from faster_whisper import WhisperModel
from urllib.parse import urlparse

from langchain_classic.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader, 
    UnstructuredPowerPointLoader, 
    WebBaseLoader,
    WikipediaLoader
)
from langchain_core.documents import Document

class Loader:
    def __init__(self):
        self.files_map = {
            "pdf": PyPDFLoader,
            "docx": Docx2txtLoader,
            "txt": TextLoader,
            "md": TextLoader,
            "pptx": UnstructuredPowerPointLoader,
            "ppt": UnstructuredPowerPointLoader,
            "py": TextLoader,
            "js": TextLoader,
            "html": TextLoader,
            "cpp": TextLoader
        }
        
        self.audio_model = WhisperModel("base", device="cpu", compute_type="int8")        
    
    def _is_url(self, path):
        try:
            result = urlparse(path)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def load_docs(self, inputs, wiki_queries=None):
        all_docs = []
        
        for item in inputs:
            if self._is_url(item):
                docs = WebBaseLoader(item).load()
                all_docs.extend(docs)
                continue
                
            file_type = item.split('.')[-1].lower()
            
            if file_type in ["wav", "mp3"]:
                segments, info = self.audio_model.transcribe(item)                
                text = ''
                for segment in segments:
                    text += segment.text + " "
                doc = Document(page_content=text.strip(), metadata={"source": item, "type": "audio"})
                all_docs.append(doc)
                
            elif file_type in self.files_map:
                load_fn = self.files_map[file_type]
                docs = load_fn(item).load()
                all_docs.extend(docs)
                
            else:
                print(f"Unsupported file type: {file_type}")

        if wiki_queries:
            for query in wiki_queries:
                docs = WikipediaLoader(query=query, load_max_docs=2).load()
                all_docs.extend(docs)
                
        return all_docs
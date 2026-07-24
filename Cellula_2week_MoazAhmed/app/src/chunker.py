from langchain_classic.text_splitter import TokenTextSplitter
from langchain_core.documents import Document
class TextChunker:
    
    def chunk_text(self, text: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
        splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(text)
        return chunks
        
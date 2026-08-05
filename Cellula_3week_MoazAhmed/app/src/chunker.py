from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
class Chunker:
    def chunk_text(self, text: list[Document], chunk_size: int, chunk_overlap: int) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(text)
        return chunks
        
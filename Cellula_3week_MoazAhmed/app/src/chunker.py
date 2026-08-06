from langchain_classic.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain_core.documents import Document
class Chunker:
    def chunk_text(self, text: list[Document], chunk_size: int = 1500, chunk_overlap: int = 200) -> list[Document]:
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, 
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(text)
        return chunks
        
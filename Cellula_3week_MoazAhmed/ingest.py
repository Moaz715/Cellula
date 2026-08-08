# ingest.py
import os
from src.loader import DataLoader
from src.chunker import Chunker
from src.embedder import VectorEmbedder
from src.vectorStore import ChromaStore

def run_ingestion():
    print("=" * 50)
    print("STARTING OFFLINE INGESTION PIPELINE")
    print("=" * 50 + "\n")

    # 1. Load raw dataset
    print("[1/4] Fetching dataset from Hugging Face...")
    # Replace with your actual dataset name or path
    raw_documents = DataLoader.load()
    print(f" -> Loaded {len(raw_documents)} raw documents.\n")

    # 2. Chunk documents into smaller blocks
    print("[2/4] Chunking documents into code blocks...")
    chunker = Chunker()
    chunks = chunker.chunk_text(raw_documents, chunk_size=1000, chunk_overlap=200)
    print(f" -> Created {len(chunks)} code chunks.\n")

    # 3. Load embedding model
    print("[3/4] Initializing local BGE embedding model...")
    embedder = VectorEmbedder()
    embedding_model = embedder.get_embedding_model()
    print(" -> Embedding model ready.\n")

    # 4. Generate vectors and persist to ChromaDB
    print("[4/4] Building ChromaDB store at './chroma_data'...")
    store = ChromaStore(embedding_model=embedding_model)
    store.build(chunks=chunks)
    
    print("\n" + "=" * 50)
    print("SUCCESS: Ingestion complete! Folder './chroma_data' is ready.")
    print("=" * 50)

if __name__ == "__main__":
    run_ingestion()
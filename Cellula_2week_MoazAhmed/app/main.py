from pathlib import Path
from src.loader import DocumentLoader
from src.chunker import TextChunker
from src.embedder import VectorEmbedder
from src.vectorStore import VectorStoreManager
from src.generator import ResponseGenerator

def main():
    # ==========================================
    # 1. INITIALIZATION PHASE (Runs exactly once)
    # ==========================================
    BASE_DIR = Path(__file__).resolve().parent
    file_path = str(BASE_DIR / "data" / "numpy_docs.txt")
    
    print("Loading and chunking document...")
    loader = DocumentLoader()
    docs = loader.load_data(file_path)
    
    chunker = TextChunker()
    chunks = chunker.chunk_text(docs, chunk_size=256, chunk_overlap=30)
    
    print("Building Vector Store (this may take time)...")
    embedder = VectorEmbedder()
    embedding_model = embedder.get_embedding_model()
    vector_manager = VectorStoreManager(chunks, embedding_model)
    
    generator = ResponseGenerator()

    print("\n" + "="*50)
    print("System Ready. Type 'exit' or 'quit' to end.")
    print("="*50)

    # ==========================================
    # 2. CHAT PHASE (Loops continuously)
    # ==========================================
    while True:
        # Get user input
        question = input("\nYou: ")
        
        # Check for exit commands
        if question.lower() in ['exit', 'quit', 'q']:
            print("Shutting down chatbot. Goodbye!")
            break
            
        if not question.strip():
            continue
            
        # Retrieve and Generate
        context = vector_manager.similarity_search(question, k=3)
        answer = generator.generate_answer(question, context)
        
        # Print output
        print(f"\nBot: {answer}")

if __name__ == "__main__":
    main()
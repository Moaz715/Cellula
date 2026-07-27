from pathlib import Path
from src.loader import DocumentLoader
from src.chunker import TextChunker
from src.embedder import VectorEmbedder
from src.vectorStore import VectorStoreManager
from src.generator import ResponseGenerator

def main():
  
    BASE_DIR = Path(__file__).resolve().parent
    file_path = str(BASE_DIR / "data" / "numpy_docs.txt")
    
    print("Loading and chunking document")
    loader = DocumentLoader()
    docs = loader.load_data(file_path)
    
    chunker = TextChunker()
    chunks = chunker.chunk_text(docs, chunk_size=256, chunk_overlap=30)
    
    print("Building Vector Store")
    embedder = VectorEmbedder()
    embedding_model = embedder.get_embedding_model()
    vector_manager = VectorStoreManager(chunks, embedding_model)
    
    generator = ResponseGenerator()

    print("\n" + "="*50)
    print("System Ready. Type 'exit' or 'quit' to end.")
    print("="*50)

    while True:
       
        question = input("\nYou: ")
        
       
        if question.lower() in ['exit', 'quit', 'q']:
            print("Shutting down chatbot. Goodbye!")
            break
            
        if not question.strip():
            continue
            
        
        context = vector_manager.similarity_search(question, k=3)
        answer = generator.generate_answer(question, context)
        
        
        print(f"\nBot: {answer}")

if __name__ == "__main__":
    main()
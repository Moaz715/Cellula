from langchain_chroma import Chroma
import hashlib
class Store:
    def __init__(self, embedder_model):
        self.db = Chroma(
            collection_name="rag_collection",
            embedding_function=embedder_model,
            persist_directory="./chroma_storage"
        )
        
    def add_chunks(self, chunked_docs):
        ids = []
        for doc in chunked_docs:
            content = doc.page_content
            doc_id = hashlib.md5(content.encode('utf-8')).hexdigest()
            ids.append(doc_id)
            
        # Check which of these IDs already exist in Chroma
        existing_data = self.db.get(ids=ids)
        existing_ids = set(existing_data.get("ids", []))
        
        # Filter down to only the truly new chunks
        new_docs = []
        new_ids = []
        for i, doc_id in enumerate(ids):
            if doc_id not in existing_ids:
                new_docs.append(chunked_docs[i])
                new_ids.append(doc_id)
                
        # Print the exact breakdown to the terminal
        print("\n--- Database Insertion Report ---")
        print(f"Total chunks processed: {len(ids)}")
        print(f"Duplicate chunks ignored: {len(existing_ids)}")
        print(f"New chunks added to DB: {len(new_ids)}")
        print("---------------------------------\n")
            
        # Only process if there is actually new data
        if new_docs:
            self.db.add_documents(documents=new_docs, ids=new_ids)
        
    def search(self, query: str, k: int = 4):
        docs = self.db.similarity_search(query, k=k)
        
        context_chunks = [doc.page_content for doc in docs]
        return context_chunks
# main.py
import os
import tempfile
from typing import List
from fastapi import FastAPI, UploadFile, File 
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from schemas import UserInput, Response
from typing import List
from src.loader import Loader
from src.chunker import Chunker
from src.embedder import Embedder
from src.vectorStore import Store
from src.generator import Generator
from src.evaluator import Evaluator

app = FastAPI(title="Evaluator-Generator System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

loader = Loader()
chunker = Chunker(chunk_size=1000, chunk_overlap=150)
embedder = Embedder()
vector_store = Store(embedder_model=embedder.get_embedding_model())
generator = Generator()
evaluator = Evaluator()


@app.post('/api/upload')
async def upload_files(files: List[UploadFile] = File(...)):
    file_paths = []
    
    try:
        for uploaded_file in files:
            ext = uploaded_file.filename.split('.')[-1]
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}")
            
            content = await uploaded_file.read()
            temp_file.write(content)
            temp_file.close()
            file_paths.append(temp_file.name)
            
        raw_docs = loader.load_docs(file_paths)
        chunked_docs = chunker.split_docs(raw_docs)
        vector_store.add_chunks(chunked_docs)
        
        return {
            "status": "success", 
            "message": f"Successfully processed and embedded {len(files)} file(s) into {len(chunked_docs)} chunks."
        }
        
    finally:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)


@app.post('/api/chat')
def chat(req: UserInput):
    """Handles vector search and the Generator-Evaluator loop."""
    user_query = req.query
    
    # 1. Retrieve Context
    context_chunks = vector_store.search(user_query, k=4)
    
    # 2. Generator-Evaluator Loop
    max_retries = 4
    feedback = "None."
    final_answer = ""
    
    for attempt in range(max_retries):
        print(f"\n--- Generation Attempt {attempt + 1} ---")
        
        # Generate Draft
        draft_answer = generator.generate_answer(user_query, context_chunks, feedback=feedback)
            
        # Evaluate Draft
        eval_result = evaluator.evaluate_answer(user_query, draft_answer, context_chunks)
        print(f"Evaluator Verdict: {eval_result}")
        
        # Check Condition
        if "STATUS: PASS" in eval_result:
            final_answer = draft_answer
            generator.save_to_memory(user_query, final_answer)
            evaluator.save_to_memory(user_query, final_answer)
            break
        else:
            feedback = eval_result
            final_answer = draft_answer # Fallback to latest draft if retries run out
            
    # 3. Return Final Answer (MOVED OUTSIDE THE FOR LOOP!)
    return StreamingResponse(iter([final_answer]), media_type="text/plain")
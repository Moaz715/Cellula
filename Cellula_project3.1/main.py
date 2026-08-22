import os
import tempfile
import shutil
import json
import redis
import numpy as np
from typing import List
from fastapi import FastAPI, UploadFile, File 
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from schemas import UserInput, IngestRequest
from src.loader import Loader
from src.chunker import Chunker
from src.embedder import Embedder
from src.vectorStore import Store
from src.generator import Generator
from src.evaluator import Evaluator

app = FastAPI(title="Evaluator-Generator System")
redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

persist_dir = "./chroma_storage"
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir, ignore_errors=True)

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
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        for path in file_paths:
            if os.path.exists(path):
                os.remove(path)


@app.post('/api/ingest-links')
def ingest_links(req: IngestRequest):
    try:
        raw_docs = loader.load_docs(inputs=req.urls, wiki_queries=req.wiki_topics)
        if not raw_docs:
            return {"status": "warning", "message": "No content found."}
            
        chunked_docs = chunker.split_docs(raw_docs)
        vector_store.add_chunks(chunked_docs)
        
        return {"status": "success", "message": f"Embedded {len(chunked_docs)} chunks from web sources."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.post('/api/chat')
def chat(req: UserInput):
    user_query = req.query
    threshold = 0.8
    query_vector = np.array(embedder.get_embedding_model().embed_query(user_query))
    
    cached_keys = redis_client.keys("semantic_cache:*")
    best_score = 0.0
    cached_answer = None
    
    for key in cached_keys:
        try:
            data = json.loads(redis_client.get(key))
            cached_vector = np.array(data["vector"])
            
            dot_product = np.dot(query_vector, cached_vector)
            norm_a = np.linalg.norm(query_vector)
            norm_b = np.linalg.norm(cached_vector)
            similarity = dot_product / (norm_a * norm_b)
            
            if similarity > best_score:
                best_score = similarity
                cached_answer = data["answer"]
        except Exception:
            continue

    if best_score >= threshold:
        return StreamingResponse(iter([f"[Cached] {cached_answer}"]), media_type="text/plain")

        
    context_chunks = vector_store.search(user_query, k=4)
    
    max_retries = 4
    feedback = "None."
    final_answer = ""
    
    for attempt in range(max_retries):
        draft_answer = generator.generate_answer(user_query, context_chunks, feedback=feedback)
        eval_result = evaluator.evaluate_answer(user_query, draft_answer, context_chunks)
        
        if "STATUS: PASS" in eval_result:
            final_answer = draft_answer
            generator.save_to_memory(user_query, final_answer)
            evaluator.save_to_memory(user_query, final_answer)
            break
        else:
            feedback = eval_result
            final_answer = draft_answer 

    negative_phrases = ["i cannot answer", "does not contain", "no relevant material"]
    is_negative = any(phrase in final_answer.lower() for phrase in negative_phrases)

    if not is_negative and final_answer:
        cache_data = json.dumps({
            "vector": query_vector.tolist(),
            "answer": final_answer
        })
        redis_client.setex(f"semantic_cache:{user_query}", 86400, cache_data)
    
    return StreamingResponse(iter([final_answer]), media_type="text/plain")
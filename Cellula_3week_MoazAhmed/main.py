import os
import sqlite3
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from dotenv import load_dotenv
load_dotenv()
d_path = os.getenv("D_PATH")
if d_path:
    os.environ['HF_HOME'] = d_path
from faster_whisper import WhisperModel

from schemas import ChatMessage, UserRequest, CodeExecutionRequest, CodeExecutionResponse, SolutionRequest
from src.embedder import VectorEmbedder
from src.vectorStore import ChromaStore
from src.grader2 import RelevanceGrader
from src.generator import ResponseGenerator
from src.intentClassifier import IntentClassifier
from src.reformulator import QueryReformulator
from src.executor import CodeExecutor


def get_live_schema(db_path: str = "app_database.db"):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';")
    tables = cursor.fetchall()
    conn.close()
    return "\n\n".join([row[0] for row in tables if row[0] is not None])

resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    embedder = VectorEmbedder()
    store = ChromaStore(embedding_model=embedder.get_embedding_model())
    store.load()
    
    resources["store"] = store
    resources["grader"] = RelevanceGrader(threshold=0.5)
    resources["intent_cls"] = IntentClassifier()
    resources["reformulator"] = QueryReformulator()
    resources["generator"] = ResponseGenerator()
    resources["whisper"] = WhisperModel("tiny.en", device="cpu", compute_type="int8")
    yield
    resources.clear()

app = FastAPI(title="AI Code Copilot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/api/chat')
async def chat(request: UserRequest):
    reformulator = resources["reformulator"]
    intent_cls = resources["intent_cls"]
    store = resources["store"]
    grader = resources["grader"]
    generator = resources["generator"]
    
    last10_history = [m.model_dump() for m in request.history[-10:]]
    new_prompt = reformulator.reformulate(request.prompt, last10_history)
    
    intent = intent_cls.classify(request.prompt)
    
    if intent == 'EXPLAIN':
        raw_stream = generator.explain_answer(new_prompt)
        return StreamingResponse(
            (chunk.content for chunk in raw_stream), 
            media_type="text/event-stream"
        )
    
    db_docs = store.vector_db.similarity_search(new_prompt, k=10)
    final_docs = []
    for doc in db_docs:
        if grader.check_relevance(new_prompt, doc.page_content):
            test_code = doc.metadata.get("test_code", "# No official tests available.")
            solution_code = doc.metadata.get("solution", "# No official solution provided.")
            combined_chunk = f"Prompt:\n{doc.page_content}\n\nSolution:\n{solution_code}\n\nTests:\n{test_code}"
            final_docs.append(combined_chunk)

    if not final_docs:
        return {
            "status": "fallback",
            "intent": intent,
            "message": "Cross-Encoder rejected all chunks. Context missing in DB.",
            "unanswered_query": new_prompt
        }

    raw_stream = generator.generate_answer(new_prompt, final_docs)
    return StreamingResponse(
        (chunk.content for chunk in raw_stream),
        media_type="text/event-stream"
    )

@app.post("/api/execute", response_model=CodeExecutionResponse)
def execute_code(request: CodeExecutionRequest):
    output, success = CodeExecutor.execute_python_code(request.code)
    return {"output": output, "success": success}
    
@app.post("/api/solution")
def save_solution(request: SolutionRequest):
    store = resources["store"]
    
    searchable_text = f"def user_solution():\n    \"\"\"{request.query}\"\"\""
    
    store.add_document(
        code_text=searchable_text, 
        metadata={
            "source": "user_contribution",
            "original_query": request.query,
            "solution": request.code.strip(),
            "test_code": "# User contribution - no dataset test available."
        }
    )
    return {"status": "success", "message": "Solution indexed into vector DB."}


@app.post("/api/voice-query")
async def voice_query(audio_file: UploadFile = File(...)):
    whisper = resources["whisper"]
    generator = resources["generator"]
    
    temp_filename = f"temp_{audio_file.filename}"
    
    
    with open(temp_filename, "wb") as buffer:
        buffer.write(await audio_file.read())
        
    try:
        segments, _ = whisper.transcribe(temp_filename, beam_size=5)
        transcript = " ".join([segment.text for segment in segments]).strip()
        
        if not transcript:
            raise HTTPException(status_code=400, detail="Audio transcript was empty.")
            
        schema = get_live_schema("app_database.db")
        sql_query = generator.generate_sql(query=transcript, schema=schema)
        
        conn = sqlite3.connect("app_database.db")
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        conn.close()
        
        results = [dict(row) for row in rows]
        
        return {
            "transcript": transcript,
            "sql_query": sql_query,
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
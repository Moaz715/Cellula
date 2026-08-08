# app.py
import os
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from dotenv import load_dotenv
import re

# 1. LOAD ENV VARS FIRST
load_dotenv()
d_path = os.getenv("D_PATH")
if d_path:
    os.environ['HF_HOME'] = d_path

import streamlit as st
from src.embedder import VectorEmbedder
from src.vectorStore import ChromaStore
from src.grader2 import RelevanceGrader
from src.generator import ResponseGenerator
from src.intentClassifier import IntentClassifier
from src.reformulator import QueryReformulator
from src.executor import CodeExecutor

st.set_page_config(page_title="Corrective RAG Copilot", layout="wide")
st.title("LLM Coding")

# 4. Load Shared Resources
@st.cache_resource
def load_shared_resources():
    embedder = VectorEmbedder()
    store = ChromaStore(embedding_model=embedder.get_embedding_model())
    store.load()
    grader = RelevanceGrader(threshold=0.5) 
    intent_cls = IntentClassifier()
    reformulator = QueryReformulator()
    return store, grader, intent_cls, reformulator

store, grader, intent_cls, reformulator = load_shared_resources()

# Sidebar Stats
st.sidebar.header("Database Info")
try:
    st.sidebar.metric("Indexed Chunks in DB", store.vector_db._collection.count())
except Exception:
    st.sidebar.write("Collection loading...")

# 5. Initialize Session State
if "generator" not in st.session_state:
    st.session_state.generator = ResponseGenerator()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_user_solution" not in st.session_state:
    st.session_state.awaiting_user_solution = False
if "unanswered_query" not in st.session_state:
    st.session_state.unanswered_query = ""

# 6. Render Chat History & Secure Execution Buttons
for idx, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # NEW: Only show button if role is assistant AND intent is GENERATE
        if msg["role"] == "assistant" and msg.get("intent") == "GENERATE" and "```python" in msg["content"]:
            if st.button("Execute Code", key=f"exec_{idx}"):
                code_match = re.search(r"```python(.*?)```", msg["content"], re.DOTALL)
                if code_match:
                    with st.spinner("Executing code..."):
                        out, success = CodeExecutor.execute_python_code(code_match.group(1))
                        if success:
                            st.success("Execution Output:")
                            st.code(out, language="text")
                        else:
                            st.error("Execution / Assertion Error:")
                            st.code(out, language="text")

# 7. Main Chat Processing Loop
if prompt := st.chat_input("Ask a coding question..."):
    st.session_state.awaiting_user_solution = False
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Step A: Reformulate Query (Preserves standalone queries)
    with st.spinner("Analyzing context..."):
        standalone_prompt = reformulator.reformulate(prompt, st.session_state.messages[:-1])
        if standalone_prompt != prompt:
            st.caption(f"*(Context applied: {standalone_prompt})*")

    # Step B: Classify Intent
    with st.spinner("Classifying intent..."):
        # CHANGED: We now pass the original `prompt` so the classifier sees action verbs!
        intent = intent_cls.classify(prompt)
        st.toast(f"Intent Detected: **{intent}**")

    # Step C: Execute Intent Path
    if intent == "EXPLAIN":
        with st.chat_message("assistant"):
            stream = st.session_state.generator.explain_answer(standalone_prompt)
            full_response = st.write_stream(stream)
            
        st.session_state.generator.save_to_memory(standalone_prompt, full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response, "intent": "EXPLAIN"})
        st.rerun()

    else:
        with st.spinner("Searching vector store (k=10)..."):
            raw_docs = store.vector_db.similarity_search(standalone_prompt, k=10)

        with st.spinner("Grading chunks with Cross-Encoder..."):
            verified_chunks = []
            for doc in raw_docs:
                if grader.check_relevance(standalone_prompt, doc.page_content):
                    
                    test_code = doc.metadata.get("test_code", "# No official tests available.")
                    
                    solution_code = doc.metadata.get("solution", "# No official solution provided.") 
                    
                    combined_chunk = (
                        f"Code Reference (Prompt):\n{doc.page_content}\n\n"
                        f"Official Solution:\n{solution_code}\n\n"
                        f"Official Tests:\n{test_code}"
                    )
                    verified_chunks.append(combined_chunk)

        if len(verified_chunks) == 0:
            st.session_state.awaiting_user_solution = True
            st.session_state.unanswered_query = standalone_prompt
            
            warning_msg = f"Cross-Encoder rejected all {len(raw_docs)} retrieved chunks. Context missing in DB."
            with st.chat_message("assistant"):
                st.warning(warning_msg)
            # Add intent here as well to prevent button rendering errors on fallbacks
            st.session_state.messages.append({"role": "assistant", "content": warning_msg, "intent": "GENERATE"})

        # Successful Retrieval Generation
        else:
            with st.chat_message("assistant"):
                st.caption(f"Verified {len(verified_chunks)} chunk(s). Generating...")
                stream = st.session_state.generator.generate_answer(standalone_prompt, verified_chunks)
                full_response = st.write_stream(stream)

            st.session_state.generator.save_to_memory(standalone_prompt, full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response, "intent": "GENERATE"})
            st.rerun()

# 8. Human-in-the-Loop Active Learning Form
if st.session_state.awaiting_user_solution:
    st.markdown("---")
    st.info(f"**Teach the System:** Provide a solution for *'{st.session_state.unanswered_query}'*:")
    user_code = st.text_area("Paste Python Code:", height=150)
    
    if st.button("Save Solution to Vector DB"):
        if user_code.strip():
            enriched_code = f"# Task: {st.session_state.unanswered_query}\n\n{user_code.strip()}"
            store.add_document(
                code_text=enriched_code, 
                metadata={
                    "source": "user_contribution", 
                    "original_query": st.session_state.unanswered_query,
                    "test_code": "# User contribution - no dataset test available."
                }
            )
            st.success("Solution saved! Run your query again.")
            st.session_state.awaiting_user_solution = False
            st.rerun()
        else:
            st.error("Please enter valid code before saving.")
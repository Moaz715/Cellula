# app.py
import re
import requests
import streamlit as st
import pandas as pd

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Multi-Modal Copilot", layout="wide")
st.title("AI Code & Database Copilot")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "awaiting_user_solution" not in st.session_state:
    st.session_state.awaiting_user_solution = False
if "unanswered_query" not in st.session_state:
    st.session_state.unanswered_query = ""

tab1, tab2 = st.tabs(["LLM Coding", "Speech to text"])


with tab1:
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
            if msg["role"] == "assistant" and "```python" in msg["content"]:
                if st.button("Execute Code", key=f"exec_{idx}"):
                    code_match = re.search(r"```python(.*?)```", msg["content"], re.DOTALL)
                    if code_match:
                        code_to_run = code_match.group(1).strip()
                        with st.spinner("Running code on backend sandbox..."):
                            try:
                                res = requests.post(
                                    f"{API_BASE_URL}/api/execute",
                                    json={"code": code_to_run},
                                    timeout=15
                                )
                                result = res.json()
                                if result.get("success"):
                                    st.success("Execution Output:")
                                    st.code(result.get("output"), language="text")
                                else:
                                    st.error("Execution Error:")
                                    st.code(result.get("output"), language="text")
                            except Exception as e:
                                st.error(f"Failed to connect to backend: {e}")

    if prompt := st.chat_input("Ask a coding question or explain a concept..."):
        st.session_state.awaiting_user_solution = False
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)

        payload = {
            "prompt": prompt,
            "history": st.session_state.messages[:-1]
        }

        with st.chat_message("assistant"):
            try:
                response = requests.post(
                    f"{API_BASE_URL}/api/chat",
                    json=payload,
                    stream=True,
                    timeout=60
                )

                content_type = response.headers.get("content-type", "")

                if "application/json" in content_type:
                    fallback_data = response.json()
                    warning_msg = fallback_data.get("message", "Context missing in vector DB.")
                    st.warning(warning_msg)
                    
                    st.session_state.awaiting_user_solution = True
                    st.session_state.unanswered_query = fallback_data.get("unanswered_query", prompt)
                    st.session_state.messages.append({"role": "assistant", "content": warning_msg})
                else:
                    def token_stream():
                        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                            if chunk:
                                yield chunk

                    full_response = st.write_stream(token_stream())
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                    st.rerun()

            except Exception as e:
                st.error(f"Error communicating with backend: {e}")

    if st.session_state.awaiting_user_solution:
        st.markdown("---")
        st.info(f"**Teach the System:** Provide a verified solution for *'{st.session_state.unanswered_query}'*:")
        user_code = st.text_area("Paste Python Code:", height=150)
        
        if st.button("Save Solution to Vector DB"):
            if user_code.strip():
                try:
                    res = requests.post(
                        f"{API_BASE_URL}/api/solution",
                        json={
                            "query": st.session_state.unanswered_query,
                            "code": user_code.strip()
                        }
                    )
                    if res.status_code == 200:
                        st.success("Solution saved into Vector DB! Re-enter your query to test.")
                        st.session_state.awaiting_user_solution = False
                        st.rerun()
                    else:
                        st.error(f"Backend error: {res.text}")
                except Exception as e:
                    st.error(f"Could not reach backend: {e}")
            else:
                st.error("Please enter valid code before saving.")



with tab2:
    st.header("Voice to SQL Database Query")
    st.markdown("Record your spoken request. The FastAPI backend will transcribe, generate SQL, and execute it on SQLite.")

    audio_value = st.audio_input("Record audio command:")

    if audio_value:
        with st.spinner("Processing speech on FastAPI backend..."):
            try:
                files = {
                    "audio_file": ("voice_command.wav", audio_value.getvalue(), "audio/wav")
                }
                
                res = requests.post(f"{API_BASE_URL}/api/voice-query", files=files, timeout=45)
                
                if res.status_code == 200:
                    data = res.json()
                    
                    st.success(f"**Transcript:** \"{data.get('transcript')}\"")
                    
                    st.info("**Generated SQL Query:**")
                    st.code(data.get("sql_query"), language="sql")
                    
                    st.write("### Database Results")
                    results_list = data.get("results", [])
                    if results_list:
                        df = pd.DataFrame(results_list)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("Query executed successfully, but returned 0 rows.")
                else:
                    st.error(f"Backend Error ({res.status_code}): {res.text}")
                    
            except Exception as e:
                st.error(f"Failed to connect to FastAPI backend: {e}")
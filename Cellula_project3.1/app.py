import streamlit as st
import requests

# Define backend URLs
BASE_URL = "http://127.0.0.1:8000"
UPLOAD_URL = f"{BASE_URL}/api/upload"
CHAT_URL = f"{BASE_URL}/api/chat"

st.set_page_config(page_title="RAG Tester", layout="wide")
st.title("Evaluator-Generator System Tester")

# --- Sidebar: File Uploads ---
with st.sidebar:
    st.header("1. Upload Documents")
    uploaded_files = st.file_uploader("Choose files to embed", accept_multiple_files=True)
    
    if st.button("Process Files"):
        if uploaded_files:
            with st.spinner("Uploading and embedding chunks..."):
                # Prepare files for the multipart/form-data request
                files_to_send = [
                    ("files", (file.name, file.getvalue(), file.type)) 
                    for file in uploaded_files
                ]
                
                try:
                    response = requests.post(UPLOAD_URL, files=files_to_send)
                    if response.status_code == 200:
                        st.success(response.json().get("message", "Success!"))
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection failed: Is FastAPI running? Error: {e}")
        else:
            st.warning("Please select files first.")

# --- Main Area: Chat Interface ---
st.header("2. Ask Questions")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your files..."):
    # Display user message instantly
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display assistant response placeholder
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Evaluating and generating answer..."):
            try:
                # Call the FastAPI chat endpoint
                response = requests.post(
                    CHAT_URL, 
                    json={"query": prompt}, 
                    stream=True
                )
                
                if response.status_code == 200:
                    # Stream the response chunks to the UI
                    for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
                        if chunk:
                            full_response += chunk
                            # Add a blinking cursor effect while typing
                            message_placeholder.markdown(full_response + "▌")
                    # Final output without the cursor
                    message_placeholder.markdown(full_response)
                else:
                    full_response = f"Backend Error: {response.status_code}"
                    message_placeholder.error(full_response)
            except Exception as e:
                full_response = f"Connection failed: {e}"
                message_placeholder.error(full_response)
        
    # Save the final response to session state
    st.session_state.messages.append({"role": "assistant", "content": full_response})
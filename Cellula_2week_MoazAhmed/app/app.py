import streamlit as st
from pathlib import Path

from src.loader import DocumentLoader
from src.chunker import TextChunker
from src.embedder import VectorEmbedder
from src.vectorStore import VectorStoreManager
from src.generator import ResponseGenerator

st.set_page_config(page_title="RAG Assistant")
st.title("Premier League Data Assistant")

@st.cache_resource(show_spinner="Initializing Knowledge Base...")
def initialize_knowledge_base():
    BASE_DIR = Path(__file__).resolve().parent
    file_path = str(BASE_DIR / "data" / "premLeague.txt")
    
    loader = DocumentLoader()
    docs = loader.load_data(file_path)
    
    chunker = TextChunker()
    chunks = chunker.chunk_text(docs, chunk_size=500, chunk_overlap=50)
    
    embedder = VectorEmbedder()
    embedding_model = embedder.get_embedding_model()
    
    vector_manager = VectorStoreManager(chunks, embedding_model)
    return vector_manager

vector_manager = initialize_knowledge_base()

if "generator" not in st.session_state:
    st.session_state.generator = ResponseGenerator()
    
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the data..."):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    CASUAL_KEYWORDS = [
        "hello", "hi", "hey", "my name is", "what is my name", 
        "who am i", "who are you", "what can you do"
    ]
    is_casual = any(kw in prompt.lower() for kw in CASUAL_KEYWORDS)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if is_casual:
                context = []
            else:
                context = vector_manager.similarity_search(prompt, k=3)
            
            response = st.session_state.generator.generate_answer(prompt, context)
            st.markdown(response)
            
    st.session_state.messages.append({"role": "assistant", "content": response})
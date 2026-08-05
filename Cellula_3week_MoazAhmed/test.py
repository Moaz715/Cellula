import streamlit as st

st.set_page_config(page_title="hoba", layout="wide")

st.title("hoba tito mambo")

if "all_documents" not in st.session_state:
    st.session_state.all_documents = []
    


column_1, column_2 = st.columns([1,3])

with column_1:
    files = st.file_uploader(label="upload here", accept_multiple_files=True, type=['txt', 'pdf', 'md', 'py', 'csv', 'xlsx', 'docx', 'pptx'])
    
    col1, col2 = st.columns([1,1])
    with col1:
        load_btn = st.button(label="Load files")
    with col2:
        clear_btn = st.button(label="clear files")
    
    if clear_btn:
        st.session_state.all_documents = []
        st.success("clear")
        st.rerun()
        
    if load_btn:
        
with column_2:
    st.title("view here")
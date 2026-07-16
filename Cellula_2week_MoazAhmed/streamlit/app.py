import streamlit as st
import requests

# The URL where your FastAPI server is running locally
FASTAPI_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="Safety Moderator Dashboard", layout="centered")
st.title("🛡️ Content Moderation System")
st.write("Analyze text queries or uploaded images through the FastAPI backend.")

tab1, tab2 = st.tabs(["💬 Text Classification", "🖼️ Image Classification"])

with tab1:
    st.subheader("Text Query Moderation")
    user_text = st.text_area("Enter text query to analyze:", height=120, placeholder="Type something here...")
    
    if st.button("Classify Text", key="btn_text"):
        if not user_text.strip():
            st.warning("Please enter some text first.")
        else:
            with st.spinner("Sending text query to FastAPI..."):
                try:
                    payload = {"text": user_text}
                    response = requests.post(f"{FASTAPI_URL}/api/classify", json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        category = result.get("prediction", "unknown")
                        
                        st.markdown("### Result:")
                        if category == "safe":
                            st.success(f"**Classification:** {category.upper()}")
                        else:
                            st.error(f"**Classification:** {category.upper()}")
                    else:
                        st.error(f"Backend Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")

with tab2:
    st.subheader("Image Content Moderation")
    uploaded_file = st.file_uploader("Upload an image (JPG, PNG, JPEG):", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image Preview", use_container_width=True)
        
        if st.button("Classify Image", key="btn_image"):
            with st.spinner("Extracting caption and analyzing via FastAPI..."):
                try:
                    files = {
                        "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                    }
                    
                    response = requests.post(f"{FASTAPI_URL}/api/classifyImage", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        extracted_caption = result.get("text", "")
                        category = result.get("prediction", "unknown")
                        
                        st.markdown("### Result:")
                        st.info(f"**Generated Image Caption (BLIP1):** {extracted_caption}")
                        
                        if category == "safe":
                            st.success(f"**Safety Classification:** {category.upper()}")
                        else:
                            st.error(f"**Safety Classification:** {category.upper()}")
                    else:
                        st.error(f"Backend Error ({response.status_code}): {response.text}")
                except Exception as e:
                    st.error(f"Failed to connect to backend: {e}")
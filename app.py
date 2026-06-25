import streamlit as st
import requests
import tempfile
import os
from core.engine import process_pdf 
# Notice we no longer import generate_chat_stream here!

# Define your FastAPI endpoint URL
BACKEND_URL = "http://localhost:8000/chat/stream"

st.title("School Handbook Assistant")

# --- File Upload Sidebar ---
with st.sidebar:
    st.header("Upload Context")
    uploaded_file = st.file_uploader("Upload Handbook (PDF)", type="pdf")
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            process_pdf(tmp.name) # Keep this as a direct call for simplicity, or make a separate API endpoint for uploads
            st.success("PDF embedded successfully!")
        os.unlink(tmp.name)

# --- Chat History Setup ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input & API Streaming ---
if prompt := st.chat_input("Ask about the handbook..."):
    # 1. Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Stream assistant response from FastAPI
    with st.chat_message("assistant"):
        
        # Generator function to consume the SSE from FastAPI
        def stream_from_api():
            # Make a streaming POST request to the backend
            response = requests.post(
                BACKEND_URL,
                json={"message": prompt},
                stream=True
            )
            
            # Iterate through the Server-Sent Events
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    # Strip the "data: " prefix from the SSE format we set up in api.py
                    if decoded_line.startswith("data: "):
                        yield decoded_line[6:]

        # Streamlit's write_stream takes our generator and animates the text
        response_text = st.write_stream(stream_from_api())
        
    # 3. Save to Streamlit history
    st.session_state.messages.append({"role": "assistant", "content": response_text})
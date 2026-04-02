import streamlit as st
import requests

# URL till din backend på Render
BACKEND_URL = "https://ai-support-bot-backend.onrender.com"

st.title("🤖 AI Customer Support Chat")

# Upload document
file = st.file_uploader("Upload a .txt file", type=["txt"])
if file:
    with st.spinner("Uploading..."):
        resp = requests.post(f"{BACKEND_URL}/upload", files={"file": file})
        user_id = resp.json().get("user_id")
        st.session_state["user_id"] = user_id
        st.success(f"Uploaded! Your session id: {user_id}")

# Chat
if "user_id" in st.session_state:
    question = st.chat_input("Ask a question")
    if question:
        with st.spinner("AI is thinking..."):
            resp = requests.post(f"{BACKEND_URL}/chat", json={
                "user_id": st.session_state["user_id"],
                "question": question
            })
            data = resp.json()
            st.chat_message("assistant").write(data.get("answer"))
            with st.expander("Sources"):
                for s in data.get("sources", []):
                    st.write(s[:300])
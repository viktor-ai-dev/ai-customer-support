from fastapi import FastAPI, UploadFile, File, Form, Header
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import re

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials")

chat_memory = {}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-support-frontend-9qcm.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    question: str

def get_user_from_token(authorization: str):
    if not authorization:
        raise ValueError("Missing Authorization header")

    token = authorization.replace("Bearer ", "")

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    supabase.auth.set_session(token, token)

    user = supabase.auth.get_user()

    if not user or not user.user:
        raise ValueError("Invalid user")

    return supabase, user.user.id

def score_doc(doc, question: str):
    words = question.lower().split()
    content = doc.page_content.lower()
    return sum(1 for w in words if re.search(rf"\b{w}\b", content))

# --------------------
# UPLOAD
# --------------------
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    doc_type: str = Form(...),
    authorization: str = Header(None)
):
    try:
        supabase, user_id = get_user_from_token(authorization)

        content = await file.read()
        text = content.decode("utf-8")

        splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        chunks = splitter.split_text(text) or [text]

        embeddings = OpenAIEmbeddings()

        Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name=user_id,
            persist_directory=f"./chroma_db/{user_id}",
            metadatas=[{"doc_type": doc_type} for _ in chunks]
        )

        supabase.table("users_docs").insert({
            "user_id": user_id,
            "collection_name": user_id,
            "doc_type": doc_type
        }).execute()

        return {"status": "uploaded"}

    except Exception as e:
        print("UPLOAD ERROR:", str(e))
        return {"error": str(e)}

# --------------------
# CHAT
# --------------------
@app.post("/chat")
async def chat(req: ChatRequest, authorization: str = Header(None)):
    try:
        supabase, user_id = get_user_from_token(authorization)

        if user_id not in chat_memory:
            chat_memory[user_id] = []

        history = chat_memory[user_id]

        history_text = "\n".join([
            f"User: {h['q']}\nAI: {h['a']}"
            for h in history[-5:]
        ])

        rewrite_llm = ChatOpenAI(model="gpt-4o-mini")

        rewritten = rewrite_llm.invoke(f"""
        Rewrite the question clearly.

        History:
        {history_text}

        Question:
        {req.question}
        """).content.strip()

        if len(rewritten) < 5:
            rewritten = req.question

        result = supabase.table("users_docs") \
            .select("*") \
            .eq("user_id", user_id) \
            .execute()

        if not result.data:
            return {"error": "No documents uploaded"}

        collection_name = result.data[-1]["collection_name"]

        embeddings = OpenAIEmbeddings()

        db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=f"./chroma_db/{collection_name}"
        )

        retriever = db.as_retriever(search_kwargs={"k": 5})

        docs = retriever.invoke(rewritten)

        context = "\n".join([doc.page_content for doc in docs])

        llm = ChatOpenAI(model="gpt-4o-mini")

        response = llm.invoke(f"""
        Answer ONLY using this context:

        {context}

        Question: {req.question}
        """)

        chat_memory[user_id].append({
            "q": req.question,
            "a": response.content
        })

        return {
            "answer": response.content,
            "sources": [doc.page_content[:300] for doc in docs]
        }

    except Exception as e:
        print("CHAT ERROR:", str(e))
        return {"error": str(e)}

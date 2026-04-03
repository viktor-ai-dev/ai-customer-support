from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import uuid
from dotenv import load_dotenv
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions

load_dotenv()

# --------------------
# Supabase
# --------------------
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")  # anon key (rekommenderas) eller service_role (test)

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase credentials")

options = ClientOptions(auto_refresh_token=True, persist_session=True)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY, options=options)

# --------------------
# App
# --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ai-support-frontend-9qcm.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Request model
# --------------------
class ChatRequest(BaseModel):
    user_id: str
    question: str

# --------------------
# Upload
# --------------------
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        text = content.decode("utf-8")

        # Split text
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_text(text) or [text]
        embeddings = OpenAIEmbeddings()

        # --------------------
        # Supabase authentication (only if using anon key)
        # --------------------
        supabase_email = os.getenv("SUPABASE_USERNAME")
        supabase_pass = os.getenv("SUPABASE_PASSWORD")

        # If we are using service_role key, skip login and use a fixed user_id for testing
        # For production with anon key, we must log in.
        use_service_role = SUPABASE_KEY.startswith("sb_secret_")  # crude check

        if use_service_role:
            # For testing with service_role – bypass RLS, use a known user_id
            print("⚠️ Using service_role key – RLS bypassed. Use only for testing.")
            # You can either take user_id from environment or generate a fixed one
            user_id = os.getenv("TEST_USER_ID", str(uuid.uuid4()))
            print(f"Using test user_id: {user_id}")
        else:
            # Using anon key – must authenticate
            if not supabase_email or not supabase_pass:
                raise ValueError("Supabase email/password missing in .env for anon key authentication")

            auth_response = supabase.auth.sign_in_with_password({
                "email": supabase_email,
                "password": supabase_pass
            })

            if not auth_response.session:
                raise ValueError("Login failed – no session returned")

            # Explicitly set session on the client
            supabase.auth.set_session(
                auth_response.session.access_token,
                auth_response.session.refresh_token
            )

            user_id = str(auth_response.user.id)
            print("Inloggad användare:", user_id)
            print("Session finns:", bool(auth_response.session))

            # Verify session is active
            try:
                user = supabase.auth.get_user()
                print("Aktiv användare (verifierad):", user.user.id)
            except Exception as e:
                print("INGEN AKTIV SESSION:", e)
                raise ValueError("Inte inloggad trots login-försök")

        # --------------------
        # Create Chroma database
        # --------------------
        db = Chroma.from_texts(
            texts=chunks,
            embedding=embeddings,
            collection_name=user_id,
            persist_directory=f"./chroma_db/{user_id}"
        )

        # --------------------
        # Save to Supabase
        # --------------------
        result = supabase.table("users_docs").insert({
            "user_id": user_id,
            "collection_name": user_id
        }).execute()

        print("Insert result:", result.data)

        return {"user_id": user_id}

    except Exception as e:
        print("UPLOAD ERROR:", str(e))
        return {"error": str(e)}

# --------------------
# Chat
# --------------------
@app.post("/chat")
async def chat(req: ChatRequest):
    try:
        print("Incoming request: ", req)

        # Get user from DB
        response = supabase.table("users_docs") \
            .select("*") \
            .eq("user_id", req.user_id) \
            .execute()
        
        print("Supabase response:", response.data)

        if not response.data:
            return {"error": "User not found"}

        collection_name = response.data[0]["collection_name"]

        # Load vector DB
        embeddings = OpenAIEmbeddings()

        db = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=f"./chroma_db/{collection_name}"
        )

        retriever = db.as_retriever(search_kwargs={"k": 2})

        # Retrieve docs
        docs = retriever.invoke(req.question)
        context = "\n".join([doc.page_content for doc in docs])

        # LLM
        llm = ChatOpenAI(model="gpt-4o-mini")

        response = llm.invoke(
            f"""You are a professional customer support AI.
                Answer ONLY using the context below.
                If the answer is not in the context, say 'I don't know'.

                Context:
                {context}

                Question: {req.question}
                Answer:"""
            )

        sources = [doc.page_content[:300] for doc in docs]

        return {
            "answer": response.content,
            "sources": sources
        }

    except Exception as e:
        print("CHAT ERROR:", str(e))
        return {"error": str(e)}
"""
Aether Research – RAG Research Assistant Backend
FastAPI + LangChain LCEL + ChromaDB + Groq
"""

import os, json, uuid, shutil, time, hashlib, base64, secrets
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timedelta

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()

# ── Auth imports ───────────────────────────────────────────────────────────
from jose import JWTError, jwt
from passlib.context import CryptContext
from cryptography.fernet import Fernet

# ── LangChain LCEL imports ────────────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="ScholarAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="public"), name="static")

# ── Paths ──────────────────────────────────────────────────────────────────────
CHROMA_DIR = Path("chroma_db")
UPLOADS_DIR = Path("uploads")
MEMORY_FILE = Path("memory.json")
SETTINGS_FILE = Path("settings.json")

CHROMA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# ── Global state ──────────────────────────────────────────────────────────────
_embeddings: Optional[HuggingFaceEmbeddings] = None
_vectorstore: Optional[Chroma] = None
_chat_history: dict = {}     # session_id → list[HumanMessage | AIMessage]
_chat_log: dict = {}         # session_id → list[{role, content, ts, sources}]

# ── Security config ───────────────────────────────────────────────────────
SECRET_KEY = os.getenv("AETHER_SECRET", "")
if not SECRET_KEY:
    SECRET_KEY = secrets.token_hex(32)
    print("⚠️  AETHER_SECRET not set. Using auto-generated key (will change on restart).")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_DAYS = 7

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*").split(",")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)
USERS_FILE = Path("users.json")

# ── Rate limiter (in-memory) ──────────────────────────────────────────────
_RATE_LIMIT: dict = {}  # ip → [timestamps]

def rate_limit(ip: str, max_req: int = 5, window: int = 60):
    now = time.time()
    timestamps = _RATE_LIMIT.get(ip, [])
    timestamps = [t for t in timestamps if now - t < window]
    _RATE_LIMIT[ip] = timestamps
    if len(timestamps) >= max_req:
        return True
    timestamps.append(now)
    return False

def load_users() -> dict:
    if USERS_FILE.exists():
        return json.loads(USERS_FILE.read_text())
    return {}

def save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2))

def create_access_token(data: dict) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[dict]:
    if credentials is None:
        return None
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    payload = verify_token(credentials)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    users = load_users()
    user = users.get(payload.get("sub"))
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return {"email": payload["sub"], "name": user["name"], "id": user["id"]}


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=get_embeddings(),
        )
    return _vectorstore


# ── Encrypted settings helpers ────────────────────────────────────────────
_SENSITIVE_KEYS = {"groq_api_key", "openai_api_key", "anthropic_api_key"}

def _fernet_key() -> bytes:
    raw = hashlib.sha256(SECRET_KEY.encode()).digest()
    return base64.urlsafe_b64encode(raw)

def _encrypt(val: str) -> str:
    if not val:
        return ""
    f = Fernet(_fernet_key())
    return f.encrypt(val.encode()).decode()

def _decrypt(val: str) -> str:
    if not val:
        return ""
    try:
        f = Fernet(_fernet_key())
        return f.decrypt(val.encode()).decode()
    except Exception:
        return None  # signal: not encrypted, needs migration


def load_settings() -> dict:
    defaults = {
        "model": "llama-3.3-70b-versatile",
        "temperature": 0.7,
        "top_p": 0.9,
        "max_tokens": 4096,
        "system_prompt": (
            "You are ScholarAI, an intelligent research assistant. "
            "Answer questions accurately using the provided context. "
            "If the context does not contain the answer, say so honestly."
        ),
        "groq_api_key": os.getenv("GROQ_API_KEY", ""),
        "openai_api_key": os.getenv("OPENAI_API_KEY", ""),
        "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY", ""),
        "user_name": "Researcher",
        "user_bio": "Exploring Retrieval-Augmented Generation & Intelligent Discovery.",
        "theme": "light",
    }
    if SETTINGS_FILE.exists():
        try:
            data = json.loads(SETTINGS_FILE.read_text())
            
            # Decrypt sensitive fields & migrate any plaintext
            migrated = False
            for k in _SENSITIVE_KEYS:
                if k in data and data[k]:
                    decrypted = _decrypt(data[k])
                    if decrypted is None:
                        data[k] = data[k]
                        migrated = True
                    else:
                        data[k] = decrypted
                else:
                    data[k] = defaults.get(k, "")
            if migrated:
                save_settings(data)

            # Auto-migrate decommissioned Groq models
            model_migrations = {
                "mixtral-8x7b-32768": "llama-3.3-70b-versatile",
                "llama3-70b-8192": "llama-3.3-70b-versatile",
                "llama3-8b-8192": "llama-3.1-8b-instant",
            }
            if data.get("model") in model_migrations:
                data["model"] = model_migrations[data["model"]]
                save_settings(data)

            for k, v in defaults.items():
                if k not in data:
                    data[k] = v
            return data
        except Exception:
            pass
    return defaults


def save_settings(data: dict):
    # Encrypt sensitive fields before saving
    out = dict(data)
    for k in _SENSITIVE_KEYS:
        if k in out and out[k]:
            out[k] = _encrypt(out[k])
    SETTINGS_FILE.write_text(json.dumps(out, indent=2))


def load_memory() -> list:
    if MEMORY_FILE.exists():
        return json.loads(MEMORY_FILE.read_text())
    return []


def save_memory(entries: list):
    MEMORY_FILE.write_text(json.dumps(entries, indent=2))


def get_llm(settings: dict):
    from langchain_groq import ChatGroq
    api_key = settings.get("groq_api_key") or os.getenv("GROQ_API_KEY", "")
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail="No Groq API key configured. Add one in API Configuration.",
        )
    return ChatGroq(
        groq_api_key=api_key,
        model_name=settings.get("model", "mixtral-8x7b-32768"),
        temperature=float(settings.get("temperature", 0.7)),
        max_tokens=int(settings.get("max_tokens", 4096)),
    )


def count_chunks() -> int:
    try:
        return get_vectorstore()._collection.count()
    except Exception:
        return 0


def list_indexed_files() -> list:
    files = []
    for p in UPLOADS_DIR.iterdir():
        if p.is_file():
            files.append({
                "name": p.name,
                "size": p.stat().st_size,
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(),
            })
    return sorted(files, key=lambda x: x["modified"], reverse=True)


# ── Pydantic models ────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str = "default"
    message: str
    model: Optional[str] = None


class SettingsUpdate(BaseModel):
    model: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    system_prompt: Optional[str] = None
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    user_name: Optional[str] = None
    user_bio: Optional[str] = None
    theme: Optional[str] = None


class MemoryEntry(BaseModel):
    title: str
    content: str
    type: str = "Concept"


class AuthRegister(BaseModel):
    email: str
    password: str
    name: str


class AuthLogin(BaseModel):
    email: str
    password: str


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/login")
def login_page():
    return FileResponse("public/login.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


@app.get("/")
def root():
    return FileResponse("public/index.html", headers={"Cache-Control": "no-cache, no-store, must-revalidate"})


# ── Auth Routes ───────────────────────────────────────────────────────────

@app.post("/api/auth/register")
def register(request: Request, data: AuthRegister):
    ip = request.client.host if request.client else "unknown"
    if rate_limit(ip, max_req=3, window=60):
        raise HTTPException(status_code=429, detail="Too many registration attempts. Try again later.")
    users = load_users()
    if data.email in users:
        raise HTTPException(status_code=400, detail="Email already registered")
    if len(data.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")
    user_id = str(uuid.uuid4())
    users[data.email] = {
        "id": user_id,
        "email": data.email,
        "name": data.name,
        "password": pwd_context.hash(data.password),
        "created": datetime.now().isoformat(),
    }
    save_users(users)
    token = create_access_token({"sub": data.email})
    return {"token": token, "user": {"id": user_id, "email": data.email, "name": data.name}}


@app.post("/api/auth/login")
def login(request: Request, data: AuthLogin):
    ip = request.client.host if request.client else "unknown"
    if rate_limit(ip, max_req=5, window=60):
        raise HTTPException(status_code=429, detail="Too many login attempts. Try again later.")
    users = load_users()
    user = users.get(data.email)
    if not user or not pwd_context.verify(data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"sub": data.email})
    return {"token": token, "user": {"id": user["id"], "email": data.email, "name": user["name"]}}


@app.get("/api/auth/me")
def get_me(current_user: dict = Depends(get_current_user)):
    return current_user


@app.get("/api/status")
def get_status():
    settings = load_settings()
    return {
        "status": "ready",
        "indexed_chunks": count_chunks(),
        "active_model": settings.get("model", "mixtral-8x7b-32768"),
        "files": list_indexed_files(),
    }


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
def chat(req: ChatRequest):
    settings = load_settings()
    if req.model:
        settings["model"] = req.model

    try:
        llm = get_llm(settings)
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})

    sys_prompt = settings.get(
        "system_prompt",
        "You are Aether, an intelligent research assistant. Answer questions based on the provided context."
    )

    # Build LCEL RAG chain
    vs = get_vectorstore()
    retriever = vs.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    prompt = ChatPromptTemplate.from_messages([
        ("system", sys_prompt + "\n\nContext from knowledge base:\n{context}"),
        MessagesPlaceholder("history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # Get or init history
    if req.session_id not in _chat_history:
        _chat_history[req.session_id] = []
    history = _chat_history[req.session_id]

    # Retrieve docs
    try:
        docs = retriever.invoke(req.message)
        context = format_docs(docs)
        sources = list({Path(d.metadata.get("source", "")).name for d in docs if d.metadata.get("source")})
    except Exception:
        context = "No documents indexed."
        sources = []
        docs = []

    # Build and run chain
    chain = prompt | llm | StrOutputParser()
    try:
        answer = chain.invoke({
            "context": context,
            "history": history,
            "question": req.message,
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": f"LLM error: {e}"})

    # Update history
    history.append(HumanMessage(content=req.message))
    history.append(AIMessage(content=answer))
    if len(history) > 20:
        history = history[-20:]
    _chat_history[req.session_id] = history

    # Store readable log
    if req.session_id not in _chat_log:
        _chat_log[req.session_id] = []
    ts = datetime.now().isoformat()
    _chat_log[req.session_id].append({"role": "user", "content": req.message, "ts": ts})
    _chat_log[req.session_id].append({"role": "assistant", "content": answer, "sources": sources, "ts": ts})

    return {
        "answer": answer,
        "sources": sources,
        "session_id": req.session_id,
        "turns": len(_chat_log[req.session_id]) // 2,
    }


@app.get("/api/chat/{session_id}/history")
def get_history(session_id: str = "default"):
    return {"history": _chat_log.get(session_id, [])}


@app.delete("/api/chat/{session_id}")
def clear_session(session_id: str = "default"):
    _chat_history.pop(session_id, None)
    _chat_log.pop(session_id, None)
    return {"message": "Session cleared"}


# ── Documents ─────────────────────────────────────────────────────────────────

@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    allowed = {".pdf", ".txt", ".md"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"File type {ext} not supported.")

    dest = UPLOADS_DIR / file.filename
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        if ext == ".pdf":
            loader = PyPDFLoader(str(dest))
        else:
            loader = TextLoader(str(dest), encoding="utf-8")
        docs = loader.load()
    except Exception as e:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to parse file: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    vs = get_vectorstore()
    vs.add_documents(chunks)
    vs.persist()

    return {
        "filename": file.filename,
        "chunks_added": len(chunks),
        "total_chunks": count_chunks(),
    }


@app.get("/api/documents")
def list_documents():
    return {"files": list_indexed_files(), "total_chunks": count_chunks()}


@app.delete("/api/documents/{filename}")
def delete_document(filename: str):
    dest = UPLOADS_DIR / filename
    if not dest.exists():
        raise HTTPException(status_code=404, detail="File not found")
    dest.unlink()

    try:
        vs = get_vectorstore()
        # Delete from Chroma using metadata filter
        vs.delete(where={"source": str(dest)})
    except Exception as e:
        print(f"Failed to delete '{filename}' chunks from vectorstore: {e}")

    return {"message": f"'{filename}' deleted and index updated"}


# ── Settings ──────────────────────────────────────────────────────────────────

@app.get("/api/settings")
def get_settings_route():
    s = load_settings()
    def mask(k):
        if k and len(k) > 8:
            return k[:4] + "•" * (len(k) - 8) + k[-4:]
        return k
    return {
        **s,
        "groq_api_key_masked": mask(s.get("groq_api_key", "")),
        "openai_api_key_masked": mask(s.get("openai_api_key", "")),
        "anthropic_api_key_masked": mask(s.get("anthropic_api_key", "")),
    }


@app.put("/api/settings")
def update_settings(data: SettingsUpdate):
    s = load_settings()
    update = data.model_dump(exclude_none=True)
    s.update(update)
    save_settings(s)
    # Clear chat histories so new model takes effect
    _chat_history.clear()
    return {"message": "Settings saved", "model": s.get("model")}


@app.post("/api/settings/test")
def test_connectivity():
    s = load_settings()
    results = {}
    key = s.get("groq_api_key") or os.getenv("GROQ_API_KEY", "")
    if key:
        try:
            from langchain_groq import ChatGroq
            llm = ChatGroq(groq_api_key=key, model_name="llama3-8b-8192", max_tokens=5)
            llm.invoke("hi")
            results["groq"] = "connected"
        except Exception as e:
            results["groq"] = f"error: {str(e)[:80]}"
    else:
        results["groq"] = "not configured"
    results["embeddings"] = "ok (local MiniLM)"
    results["vectorstore"] = f"ok ({count_chunks()} chunks indexed)"
    return results


# ── Memory ────────────────────────────────────────────────────────────────────

@app.get("/api/memory")
def get_memory():
    entries = load_memory()
    return {"entries": entries, "total": len(entries)}


@app.post("/api/memory")
def add_memory(entry: MemoryEntry):
    entries = load_memory()
    new_entry = {
        "id": str(uuid.uuid4()),
        "title": entry.title,
        "content": entry.content,
        "type": entry.type,
        "created": datetime.now().isoformat(),
    }
    entries.insert(0, new_entry)
    save_memory(entries)
    return new_entry


@app.delete("/api/memory/{entry_id}")
def delete_memory_entry(entry_id: str):
    entries = [e for e in load_memory() if e.get("id") != entry_id]
    save_memory(entries)
    return {"message": "Entry deleted"}


@app.delete("/api/memory")
def flush_memory():
    save_memory([])
    return {"message": "Memory flushed"}


# ── Models ────────────────────────────────────────────────────────────────────

MODELS = [
    {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B", "provider": "Groq", "tags": ["Reasoning", "Versatile"], "recommended": True},
    {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B", "provider": "Groq", "tags": ["Fast", "Instant"], "recommended": False},
    {"id": "gemma2-9b-it", "name": "Gemma 2 9B", "provider": "Groq", "tags": ["Accurate", "Editorial"], "recommended": False},
]


@app.get("/api/models")
def list_models():
    return {"models": MODELS}


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8001"))
    uvicorn.run(app, host="0.0.0.0", port=port)

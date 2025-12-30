import sys
import io

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
import shutil
from pathlib import Path
from backend.rag_engine import RAGEngine
from backend.database import Database
from backend.file_processor import process_uploaded_file, process_folder, extract_text_from_file
from datetime import datetime
import uuid
import zipfile
import tempfile
import asyncio
from collections import Counter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

app = FastAPI(title="Advanced RAG Chatbot API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db: Optional[Database] = None
rag_engines: Dict[str, RAGEngine] = {}
session_rag_engines: Dict[str, RAGEngine] = {}
shared_models: Dict[str, Any] = {}

@app.on_event("startup")
async def startup_event():
    global db
    print("\n" + "="*60)
    print("Starting RAG Chatbot Backend...")
    print("="*60)
    print("\n[1/3] Connecting to MongoDB...")
    max_retries = 3
    retry_delay = 1
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    if mongodb_uri.startswith("mongodb+srv://"):
        print(f"  Using MongoDB Atlas connection")
    else:
        print(f"  Connection string: {mongodb_uri}")
    for attempt in range(max_retries):
        try:
            db = Database()
            db.client.admin.command('ping')
            print("✓ MongoDB connected successfully.")
            break
        except Exception as e:
            error_msg = str(e)
            if attempt < max_retries - 1:
                print(f"  Retrying MongoDB connection (attempt {attempt + 1}/{max_retries})...")
                await asyncio.sleep(retry_delay)
            else:
                print(f"\n⚠ Warning: Could not connect to MongoDB: {error_msg}")
                print("  The service will start but database features will not work.")
                db = None
    print("\n[2/3] Initializing RAG engine...")
    try:
        base_engine = get_rag_engine(session_id=None)
        shared_models['llm_provider'] = base_engine.llm_provider
        shared_models['embedding_model'] = base_engine.embedding_model
        shared_models['reranker'] = base_engine.reranker
        shared_models['intent_classifier'] = base_engine.intent_classifier
        shared_models['use_local_llm'] = base_engine.use_local_llm
        print("✓ RAG engine initialized successfully.")
    except Exception as e:
        error_msg = str(e)
        print(f"⚠ Warning: Could not initialize RAG engine: {error_msg}")
        shared_models.clear()
    print("\n[3/3] Startup complete!")
    print("="*60)
    print("Backend is ready to accept requests.")
    port = os.getenv("PORT", "8080")
    print(f"API available at: http://0.0.0.0:{port}")
    print(f"API docs available at: http://0.0.0.0:{port}/docs")
    print("="*60 + "\n")

def get_rag_engine(session_id: Optional[str] = None) -> RAGEngine:
    use_local_env = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    openai_api_key = os.getenv("OPENAI_API_KEY")
    base_kb_folder = os.getenv("KB_FOLDER", "KB")
    model_path = "models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
    local_model_exists = os.path.exists(model_path)
    if use_local_env:
        use_local = True
    elif not openai_api_key and local_model_exists:
        use_local = True
        print("WARNING: OPENAI_API_KEY not set, but local model found. Using local model.")
    elif not openai_api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set and no local model found."
        )
    else:
        use_local = False
    if session_id:
        session_kb_folder = Path(base_kb_folder) / "sessions" / session_id
        session_index_path = f"index.faiss/sessions/{session_id}"
        session_kb_folder.mkdir(parents=True, exist_ok=True)
        if not any(session_kb_folder.iterdir()):
            base_kb_path = Path(base_kb_folder)
            if base_kb_path.exists():
                supported_extensions = {'.txt', '.md', '.markdown', '.pdf'}
                for item in base_kb_path.iterdir():
                    if item.is_file() and item.suffix.lower() in supported_extensions:
                        shutil.copy2(item, session_kb_folder / item.name)
        index_dir = Path(session_index_path).parent
        index_dir.mkdir(parents=True, exist_ok=True)
        if session_id in session_rag_engines:
            return session_rag_engines[session_id]
        if use_local and not local_model_exists:
            raise ValueError(
                f"Local model file not found at {model_path}."
            )
        if shared_models:
            session_engine = RAGEngine(
                kb_folder=str(session_kb_folder),
                index_path=session_index_path,
                use_local_llm=use_local,
                openai_api_key=openai_api_key if not use_local else None,
                shared_llm_provider=shared_models.get('llm_provider'),
                shared_embedding_model=shared_models.get('embedding_model'),
                shared_reranker=shared_models.get('reranker'),
                shared_intent_classifier=shared_models.get('intent_classifier')
            )
        else:
            session_engine = RAGEngine(
                kb_folder=str(session_kb_folder),
                index_path=session_index_path,
                use_local_llm=use_local,
                openai_api_key=openai_api_key if not use_local else None
            )
        session_rag_engines[session_id] = session_engine
        return session_engine
    engine_key = "local" if use_local else "openai"
    if engine_key not in rag_engines:
        if use_local and not local_model_exists:
            raise ValueError(
                f"Local model file not found at {model_path}."
            )
        if shared_models:
            rag_engines[engine_key] = RAGEngine(
                kb_folder=base_kb_folder,
                use_local_llm=use_local,
                openai_api_key=openai_api_key if not use_local else None,
                shared_llm_provider=shared_models.get('llm_provider'),
                shared_embedding_model=shared_models.get('embedding_model'),
                shared_reranker=shared_models.get('reranker'),
                shared_intent_classifier=shared_models.get('intent_classifier')
            )
        else:
            rag_engines[engine_key] = RAGEngine(
                kb_folder=base_kb_folder,
                use_local_llm=use_local,
                openai_api_key=openai_api_key if not use_local else None
            )
    return rag_engines[engine_key]

class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    max_tokens: Optional[int] = 1000
    min_score: Optional[float] = 0.5

class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    model_used: str
    sources: Optional[List[str]] = []
    chunks: Optional[List[Dict[str, Any]]] = []

class SessionResponse(BaseModel):
    session_id: str
    created_at: str
    message_count: int

class HealthResponse(BaseModel):
    status: str
    message: str
    models_available: Dict[str, bool]

class AnalyticsResponse(BaseModel):
    total_sessions: int
    total_messages: int
    sessions_today: int
    messages_today: int
    popular_queries: List[Dict]

@app.get("/", response_model=HealthResponse)
async def root():
    try:
        local_available = os.path.exists("models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
        openai_available = bool(os.getenv("OPENAI_API_KEY"))
        return HealthResponse(
            status="ok",
            message="Advanced RAG Chatbot API is running",
            models_available={"openai": openai_available,"local": local_available}
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Error: Service error")

@app.get("/health", response_model=HealthResponse)
async def health():
    try:
        openai_available = bool(os.getenv("OPENAI_API_KEY"))
        mongodb_available = db is not None if db else False
        return HealthResponse(
            status="ok",
            message="Service is ready",
            models_available={"openai": openai_available,"mongodb": mongodb_available}
        )
    except Exception as e:
        return HealthResponse(
            status="starting",
            message=f"Service is starting: {str(e)}",
            models_available={}
        )

@app.get("/chat")
async def chat_get():
    raise HTTPException(status_code=405, detail="Method Not Allowed. Use POST.")

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatMessage):
    try:
        if db is None:
            raise HTTPException(status_code=503, detail="Database not initialized.")
        user_message = request.message
        if isinstance(user_message, str):
            user_message = user_message.encode('utf-8', errors='replace').decode('utf-8')
        if not request.session_id:
            session_id = db.create_session()
        else:
            session_id = request.session_id
            if not db.get_session(session_id):
                session_id = db.create_session()
        history_messages = db.get_session_messages(session_id, limit=20)
        chat_history = []
        for msg in history_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                content = content.encode('utf-8', errors='replace').decode('utf-8')
            chat_history.append({"role": role, "content": content})
        engine = get_rag_engine(session_id=session_id)
        use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
        intent, confidence = engine.intent_classifier.predict(user_message)
        answer, sources, chunks = engine.answer(
            user_message,
            chat_history=chat_history,
            session_id=session_id,
            max_tokens=request.max_tokens,
            min_score=request.min_score
        )
        model_used = "local" if use_local else "openai"
        db.add_message(session_id, "user", user_message, metadata={"intent": intent, "confidence": float(confidence)})
        db.add_message(session_id, "assistant", answer, metadata={"model": model_used, "sources": sources, "chunks_count": len(chunks)})
        db.log_analytics("chat_message", {"session_id": session_id,"intent": intent,"confidence": float(confidence),"model": model_used,"message_length": len(user_message),"sources_count": len(sources),"chunks_count": len(chunks)})
        return ChatResponse(response=answer, session_id=session_id,intent=intent,confidence=float(confidence),model_used=model_used,sources=sources,chunks=chunks)
    except HTTPException:
        raise
    except ValueError as e:
        error_detail = str(e).encode('utf-8', errors='replace').decode('utf-8')
        safe_detail = "Error processing chat: {}".format(error_detail)
        raise HTTPException(status_code=400, detail=safe_detail)
    except Exception as e:
        import traceback
        error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
        safe_detail = "Error processing chat: {}. Check server logs for details.".format(error_msg)
        raise HTTPException(status_code=500, detail=safe_detail)

@app.post("/sessions", response_model=SessionResponse)
async def create_session():
    try:
        if db is None:
            raise HTTPException(status_code=503, detail="Database not initialized.")
        session_id = db.create_session()
        session = db.get_session(session_id)
        return SessionResponse(session_id=session_id, created_at=session["created_at"].isoformat(), message_count=0)
    except Exception:
        raise HTTPException(status_code=500, detail="Error creating session")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

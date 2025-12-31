import sys
import io
import os
import shutil
import uuid
import zipfile
import tempfile
import asyncio
from pathlib import Path
from collections import Counter
from datetime import datetime
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.rag_engine import RAGEngine
from backend.database import Database
from backend.file_processor import process_uploaded_file, process_folder, extract_text_from_file
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Set UTF-8 encoding for stdout and stderr
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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

WARM_SESSION_ID = "__warm__"


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
    for attempt in range(max_retries):
        try:
            db = Database()
            db.client.admin.command('ping')
            print("✓ MongoDB connected successfully.")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  Retrying MongoDB connection (attempt {attempt + 1}/{max_retries})...")
                await asyncio.sleep(retry_delay)
            else:
                print(f"\n⚠ Warning: Could not connect to MongoDB: {e}")
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
        print(f"⚠ Warning: Could not initialize RAG engine: {e}")
        shared_models.clear()

    print("\n[3/3] Warming default session engine...")
    try:
        engine = get_rag_engine(session_id=WARM_SESSION_ID)

        # ⚡ Force real warmup to reduce first-message latency
        engine.embedding_model.embed_query("warmup")
        if engine.vectorstore:
            engine.vectorstore.similarity_search("warmup", k=1)
        engine.intent_classifier.predict("warmup")

        print("✓ Default session engine warmed.")
    except Exception as e:
        print(f"⚠ Warm session failed: {e}")

    print("\n[3/3] Startup complete!")
    print("="*60)
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
        raise ValueError("OPENAI_API_KEY is not set and no local model found.")
    else:
        use_local = False

    # ---------- SESSION ENGINE ----------
    if session_id:
        if session_id in session_rag_engines:
            return session_rag_engines[session_id]

        # Clone warm session if available
        if WARM_SESSION_ID in session_rag_engines and session_id != WARM_SESSION_ID:
            warm_engine = session_rag_engines[WARM_SESSION_ID]

            session_kb_folder = Path(base_kb_folder) / "sessions" / session_id
            session_index_path = f"index.faiss/sessions/{session_id}"
            session_kb_folder.mkdir(parents=True, exist_ok=True)
            Path(session_index_path).parent.mkdir(parents=True, exist_ok=True)

            engine = RAGEngine(
                kb_folder=str(session_kb_folder),
                index_path=session_index_path,
                use_local_llm=warm_engine.use_local_llm,
                openai_api_key=openai_api_key if not warm_engine.use_local_llm else None,
                shared_llm_provider=warm_engine.llm_provider,
                shared_embedding_model=warm_engine.embedding_model,
                shared_reranker=warm_engine.reranker,
                shared_intent_classifier=warm_engine.intent_classifier
            )

            session_rag_engines[session_id] = engine
            return engine

        # Cold session fallback
        session_kb_folder = Path(base_kb_folder) / "sessions" / session_id
        session_index_path = f"index.faiss/sessions/{session_id}"
        session_kb_folder.mkdir(parents=True, exist_ok=True)
        Path(session_index_path).parent.mkdir(parents=True, exist_ok=True)

        engine = RAGEngine(
            kb_folder=str(session_kb_folder),
            index_path=session_index_path,
            use_local_llm=use_local,
            openai_api_key=openai_api_key if not use_local else None,
            shared_llm_provider=shared_models.get("llm_provider"),
            shared_embedding_model=shared_models.get("embedding_model"),
            shared_reranker=shared_models.get("reranker"),
            shared_intent_classifier=shared_models.get("intent_classifier")
        )

        session_rag_engines[session_id] = engine
        return engine

    # ---------- BASE ENGINE ----------
    engine_key = "local" if use_local else "openai"
    if engine_key not in rag_engines:
        rag_engines[engine_key] = RAGEngine(
            kb_folder=base_kb_folder,
            use_local_llm=use_local,
            openai_api_key=openai_api_key if not use_local else None
        )
    return rag_engines[engine_key]


# ----------- Models -----------
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


# ----------- Endpoints -----------

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
        raise HTTPException(status_code=500, detail="Service error")


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
        user_message = request.message.encode('utf-8', errors='replace').decode('utf-8')
        session_id = request.session_id or db.create_session()
        if not db.get_session(session_id):
            session_id = db.create_session()

        history_messages = db.get_session_messages(session_id, limit=20)
        chat_history = [{"role": msg.get("role", "user"), "content": msg.get("content","")} for msg in history_messages]

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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat: {str(e)}")


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


@app.get("/knowledge-base/files")
async def list_kb_files(session_id: Optional[str] = None):
    try:
        base_kb_folder = Path(os.getenv("KB_FOLDER", "KB"))
        if session_id:
            kb_folder = base_kb_folder / "sessions" / session_id
            kb_folder.mkdir(parents=True, exist_ok=True)
            if not any(kb_folder.iterdir()):
                supported_extensions = {'.txt', '.md', '.markdown', '.pdf'}
                if base_kb_folder.exists():
                    for item in base_kb_folder.iterdir():
                        if item.is_file() and item.suffix.lower() in supported_extensions:
                            shutil.copy2(item, kb_folder / item.name)
        else:
            kb_folder = base_kb_folder
        supported_extensions = {'.txt', '.md', '.markdown', '.pdf'}
        files = []
        if not kb_folder.exists():
            return {"files": [], "total": 0, "session_id": session_id}
        for root, dirs, filenames in os.walk(kb_folder):
            if not session_id and "sessions" in dirs:
                dirs.remove("sessions")
            for filename in filenames:
                file_path = Path(root) / filename
                if file_path.suffix.lower() in supported_extensions:
                    rel_path = os.path.relpath(file_path, kb_folder)
                    stat = file_path.stat()
                    files.append({
                        "filename": filename,
                        "path": rel_path,
                        "size": stat.st_size,
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "type": file_path.suffix.lower()
                    })
        return {
            "files": sorted(files, key=lambda x: x["modified"], reverse=True),
            "total": len(files),
            "session_id": session_id
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Error listing KB files")


@app.get("/knowledge-base/files/{file_path:path}")
async def get_kb_file_content(file_path: str, session_id: Optional[str] = None):
    try:
        base_kb_folder = Path(os.getenv("KB_FOLDER", "KB"))
        if session_id:
            kb_folder = base_kb_folder / "sessions" / session_id
            kb_folder.mkdir(parents=True, exist_ok=True)
            if not any(kb_folder.iterdir()):
                supported_extensions = {'.txt', '.md', '.markdown', '.pdf'}
                if base_kb_folder.exists():
                    for item in base_kb_folder.iterdir():
                        if item.is_file() and item.suffix.lower() in supported_extensions:
                            shutil.copy2(item, kb_folder / item.name)
        else:
            kb_folder = base_kb_folder
        file_full_path = kb_folder / file_path
        try:
            file_full_path.resolve().relative_to(kb_folder.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
        if not file_full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        supported_extensions = {'.txt', '.md', '.markdown', '.pdf'}
        if file_full_path.suffix.lower() not in supported_extensions:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        content = extract_text_from_file(file_full_path)
        stat = file_full_path.stat()
        return {
            "filename": file_full_path.name,
            "path": file_path,
            "content": content,
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "type": file_full_path.suffix.lower()
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Error reading file")


@app.delete("/knowledge-base/files/{file_path:path}")
async def delete_kb_file(file_path: str, session_id: Optional[str] = None):
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required for delete")
        base_kb_folder = Path(os.getenv("KB_FOLDER", "KB"))
        kb_folder = base_kb_folder / "sessions" / session_id
        kb_folder.mkdir(parents=True, exist_ok=True)
        file_full_path = kb_folder / file_path
        try:
            file_full_path.resolve().relative_to(kb_folder.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
        if not file_full_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        file_full_path.unlink()

        engine = get_rag_engine(session_id=session_id)
        documents = []
        supported_extensions = {'.txt', '.md', '.markdown', '.pdf'}
        for root, _, files in os.walk(kb_folder):
            for fname in files:
                fpath = Path(root) / fname
                if fpath.suffix.lower() in supported_extensions and fpath.exists():
                    try:
                        content = extract_text_from_file(fpath)
                        if content and content.strip():
                            rel_path = os.path.relpath(fpath, kb_folder)
                            documents.append(Document(
                                page_content=content.strip(),
                                metadata={"source": rel_path, "file_type": fpath.suffix.lower()}
                            ))
                    except Exception:
                        continue
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=engine.chunk_size,
                chunk_overlap=engine.chunk_overlap,
                length_function=len
            )
            split_docs = text_splitter.split_documents(documents)
            engine.vectorstore = FAISS.from_documents(split_docs, engine.embedding_model)
            engine.vectorstore.save_local(engine.index_path)
        else:
            engine.vectorstore = None
        session_rag_engines[session_id] = engine
        return {
            "message": f"File {file_path} deleted successfully from your session",
            "file_path": file_path,
            "session_id": session_id
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")


@app.post("/knowledge-base/upload")
async def upload_knowledge_base(
    files: List[UploadFile] = File(default=[]),
    session_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    kb_name: Optional[str] = Form(None)
):
    try:
        if db is None:
            raise HTTPException(status_code=503, detail="Database not initialized.")
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required for upload")
        supported_extensions = {'.txt', '.md', '.markdown', '.pdf', '.zip'}
        base_kb_folder = Path(os.getenv("KB_FOLDER", "KB"))
        kb_folder = base_kb_folder / "sessions" / session_id
        kb_folder.mkdir(parents=True, exist_ok=True)
        uploaded_files = []
        for f in files:
            if Path(f.filename).suffix.lower() not in supported_extensions:
                continue
            content = await f.read()
            dest = kb_folder / f.filename
            dest.write_bytes(content)
            uploaded_files.append(str(dest))
        if not uploaded_files:
            raise HTTPException(status_code=400, detail="No supported files uploaded")

        engine = get_rag_engine(session_id=session_id)
        documents = process_folder(kb_folder)
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=engine.chunk_size,
                chunk_overlap=engine.chunk_overlap,
                length_function=len
            )
            split_docs = text_splitter.split_documents(documents)
            engine.vectorstore = FAISS.from_documents(split_docs, engine.embedding_model)
            engine.vectorstore.save_local(engine.index_path)
            session_rag_engines[session_id] = engine

        return {"message": "Files uploaded successfully", "files": uploaded_files, "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading KB: {str(e)}")


@app.get("/analytics", response_model=AnalyticsResponse)
async def analytics():
    try:
        if db is None:
            raise HTTPException(status_code=503, detail="Database not initialized.")
        data = db.get_analytics_summary()
        return AnalyticsResponse(
            total_sessions=data.get("total_sessions", 0),
            total_messages=data.get("total_messages", 0),
            sessions_today=data.get("sessions_today", 0),
            messages_today=data.get("messages_today", 0),
            popular_queries=data.get("popular_queries", [])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving analytics: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

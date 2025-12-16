"""
FastAPI application for RAG Chatbot
"""
import sys
import io

# Set UTF-8 encoding for stdout and stderr to prevent Unicode errors
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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Cloud Storage frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database (will be set during startup)
db: Optional[Database] = None

# Initialize RAG engines (one per model type for base, plus one per session)
rag_engines: Dict[str, RAGEngine] = {}
session_rag_engines: Dict[str, RAGEngine] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize database connection and RAG engine on startup with retry logic"""
    global db
    print("\n" + "="*60)
    print("Starting RAG Chatbot Backend...")
    print("="*60)
    
    # Initialize database with retry logic (reduced retries for Cloud Run)
    print("\n[1/3] Connecting to MongoDB...")
    max_retries = 3  # Reduced for Cloud Run - fail fast if not configured
    retry_delay = 1
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    
    # Don't print full connection string in production (security)
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
                print("  Make sure MONGODB_URI environment variable is set correctly.")
                # Don't raise - allow service to start without DB for health checks
                db = None
    
    # Pre-initialize RAG engine (lazy load for faster startup in Cloud Run)
    print("\n[2/3] Checking RAG engine configuration...")
    try:
        # Just verify configuration, don't fully initialize (lazy load)
        openai_api_key = os.getenv("OPENAI_API_KEY")
        kb_folder = os.getenv("KB_FOLDER", "KB")
        
        if not openai_api_key:
            print("⚠ Warning: OPENAI_API_KEY not set. RAG will not work until configured.")
        else:
            print("✓ OpenAI API key configured.")
        
        if not os.path.exists(kb_folder) or not os.listdir(kb_folder):
            print(f"⚠ Warning: KB folder '{kb_folder}' is empty or missing.")
        else:
            print(f"✓ KB folder found with {len(os.listdir(kb_folder))} files.")
        
        print("  RAG engine will be initialized on first request (lazy loading).")
    except Exception as e:
        print(f"⚠ Warning: Could not verify RAG configuration: {e}")
        print("  RAG engine will be initialized on first request.")
    
    print("\n[3/3] Startup complete!")
    print("="*60)
    print("Backend is ready to accept requests.")
    port = os.getenv("PORT", "8080")
    print(f"API available at: http://0.0.0.0:{port}")
    print(f"API docs available at: http://0.0.0.0:{port}/docs")
    print("="*60 + "\n")


def get_rag_engine(session_id: Optional[str] = None) -> RAGEngine:
    """Get or create RAG engine - auto-detects local model if available
    If session_id is provided, returns a session-specific engine with its own KB folder
    """
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
            "OPENAI_API_KEY is not set and no local model found. "
            "Please either:\n"
            "1. Set OPENAI_API_KEY environment variable, or\n"
            "2. Set USE_LOCAL_LLM=true and ensure model file exists at models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
        )
    else:
        use_local = False
    
    # If session_id is provided, use session-specific KB folder and index
    if session_id:
        session_kb_folder = Path(base_kb_folder) / "sessions" / session_id
        session_index_path = f"index.faiss/sessions/{session_id}"
        
        # Create session KB folder if it doesn't exist
        session_kb_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize with default KB files if session folder is empty (new session)
        if not any(session_kb_folder.iterdir()):
            base_kb_path = Path(base_kb_folder)
            if base_kb_path.exists():
                # Copy default KB files to session folder (excluding sessions subfolder)
                supported_extensions = {'.txt', '.md', '.markdown', '.pdf'}
                for item in base_kb_path.iterdir():
                    if item.is_file() and item.suffix.lower() in supported_extensions:
                        shutil.copy2(item, session_kb_folder / item.name)
        
        # Create index directory if it doesn't exist
        index_dir = Path(session_index_path).parent
        index_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if we already have a RAG engine for this session
        if session_id in session_rag_engines:
            return session_rag_engines[session_id]
        
        # Create new session-specific RAG engine
        if use_local and not local_model_exists:
            raise ValueError(
                f"Local model file not found at {model_path}. "
                "Please download the model or set USE_LOCAL_LLM=false to use OpenAI."
            )
        
        session_engine = RAGEngine(
            kb_folder=str(session_kb_folder),
            index_path=session_index_path,
            use_local_llm=use_local,
            openai_api_key=openai_api_key if not use_local else None
        )
        session_rag_engines[session_id] = session_engine
        return session_engine
    
    # No session_id - use base/global KB folder
    engine_key = "local" if use_local else "openai"
    
    if engine_key not in rag_engines:
        if use_local and not local_model_exists:
            raise ValueError(
                f"Local model file not found at {model_path}. "
                "Please download the model or set USE_LOCAL_LLM=false to use OpenAI."
            )
        
        rag_engines[engine_key] = RAGEngine(
            kb_folder=base_kb_folder,
            use_local_llm=use_local,
            openai_api_key=openai_api_key if not use_local else None
        )
    
    return rag_engines[engine_key]


# Request/Response models
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
    """Health check endpoint"""
    try:
        local_available = os.path.exists("models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
        openai_available = bool(os.getenv("OPENAI_API_KEY"))
        
        return HealthResponse(
            status="ok",
            message="Advanced RAG Chatbot API is running",
            models_available={
                "openai": openai_available,
                "local": local_available
            }
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Error: Service error")


@app.get("/health", response_model=HealthResponse)
async def health():
    """Detailed health check - lightweight for Cloud Run"""
    try:
        # Lightweight health check - don't initialize RAG engine (too slow)
        openai_available = bool(os.getenv("OPENAI_API_KEY"))
        mongodb_available = db is not None if db else False
        
        return HealthResponse(
            status="ok",
            message="Service is ready",
            models_available={
                "openai": openai_available,
                "mongodb": mongodb_available
            }
        )
    except Exception as e:
        # Return 200 even on error to prevent Cloud Run from killing the container
        # The service might still be starting up
        return HealthResponse(
            status="starting",
            message=f"Service is starting: {str(e)}",
            models_available={}
        )


@app.get("/chat")
async def chat_get():
    """GET endpoint for /chat"""
    raise HTTPException(
        status_code=405, 
        detail="Method Not Allowed. Please use POST to send messages to /chat endpoint."
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatMessage):
    """Chat endpoint with multi-turn reasoning and session management"""
    try:
        if db is None:
            raise HTTPException(status_code=503, detail="Database not initialized. Please wait for the service to start.")
        
        # --- FORCE UTF-8 ENCODING ON USER INPUT ---
        user_message = request.message
        if isinstance(user_message, str):
            user_message = user_message.encode('utf-8', errors='replace').decode('utf-8')
        
        # Get or create session
        if not request.session_id:
            session_id = db.create_session()
        else:
            session_id = request.session_id
            if not db.get_session(session_id):
                session_id = db.create_session()
        
        # Get chat history
        history_messages = db.get_session_messages(session_id, limit=20)
        chat_history = []
        for msg in history_messages:
            # Ensure UTF-8 encoding on previous messages as well
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, str):
                content = content.encode('utf-8', errors='replace').decode('utf-8')
            chat_history.append({"role": role, "content": content})
        
        # Get RAG engine (session-specific)
        engine = get_rag_engine(session_id=session_id)
        use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
        
        # Get intent classification
        intent, confidence = engine.intent_classifier.predict(user_message)
        
        # Get answer, sources, and chunks
        answer, sources, chunks = engine.answer(
            user_message,
            chat_history=chat_history,
            session_id=session_id,
            max_tokens=request.max_tokens,
            min_score=request.min_score
        )
        
        # Save messages to database
        model_used = "local" if use_local else "openai"
        db.add_message(session_id, "user", user_message, metadata={"intent": intent, "confidence": float(confidence)})
        db.add_message(session_id, "assistant", answer, metadata={"model": model_used, "sources": sources, "chunks_count": len(chunks)})
        
        # Log analytics
        db.log_analytics("chat_message", {
            "session_id": session_id,
            "intent": intent,
            "confidence": float(confidence),
            "model": model_used,
            "message_length": len(user_message),
            "sources_count": len(sources),
            "chunks_count": len(chunks)
        })
        
        return ChatResponse(
            response=answer,
            session_id=session_id,
            intent=intent,
            confidence=float(confidence),
            model_used=model_used,
            sources=sources,
            chunks=chunks
        )
    except HTTPException:
        raise
    except ValueError as e:
        try:
            error_detail = str(e).encode('utf-8', errors='replace').decode('utf-8')
        except UnicodeEncodeError:
            try:
                error_detail = repr(e).encode('utf-8', errors='replace').decode('utf-8')
            except Exception:
                error_detail = "Error processing request"
        # Use .format() to avoid f-string encoding issues
        safe_detail = "Error processing chat: {}".format(error_detail)
        raise HTTPException(status_code=400, detail=safe_detail)
    except Exception as e:
        import traceback
        try:
            error_details = traceback.format_exc().encode('utf-8', errors='replace').decode('utf-8')
            print(f"Error in chat endpoint: {error_details}")
        except UnicodeEncodeError:
            print("Error in chat endpoint: [Unable to format error details]")
        try:
            error_msg = str(e).encode('utf-8', errors='replace').decode('utf-8')
        except UnicodeEncodeError:
            try:
                error_msg = repr(e).encode('utf-8', errors='replace').decode('utf-8')
            except Exception:
                error_msg = "Unknown error"
        # Use .format() to avoid f-string encoding issues
        safe_detail = "Error processing chat: {}. Check server logs for details.".format(error_msg)
        raise HTTPException(status_code=500, detail=safe_detail)



@app.post("/sessions", response_model=SessionResponse)
async def create_session():
    """Create a new chat session"""
    try:
        if db is None:
            raise HTTPException(status_code=503, detail="Database not initialized. Please wait for the service to start.")
        session_id = db.create_session()
        session = db.get_session(session_id)
        return SessionResponse(
            session_id=session_id,
            created_at=session["created_at"].isoformat(),
            message_count=0
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Error creating session")


@app.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """Get session information"""
    try:
        if db is None:
            raise HTTPException(status_code=503, detail="Database not initialized. Please wait for the service to start.")
        session = db.get_session(session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        return SessionResponse(
            session_id=session_id,
            created_at=session["created_at"].isoformat(),
            message_count=session["message_count"]
        )
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Error getting session")


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, limit: int = 50):
    """Get messages for a session"""
    try:
        if db is None:
            raise HTTPException(status_code=503, detail="Database not initialized. Please wait for the service to start.")
        messages = db.get_session_messages(session_id, limit=limit)
        return {
            "session_id": session_id,
            "messages": [
                {
                    "role": msg["role"],
                    "content": msg["content"],
                    "timestamp": msg["timestamp"].isoformat(),
                    "metadata": msg.get("metadata", {})
                }
                for msg in messages
            ]
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Error getting messages")


@app.post("/knowledge-base/upload")
async def upload_knowledge_base(
    files: List[UploadFile] = File(default=[]),
    session_id: Optional[str] = Form(None),
    user_id: Optional[str] = Form(None),
    kb_name: Optional[str] = Form(None)
):
    """Upload knowledge base files - session-specific, replaces the session's KB"""
    try:
        if db is None:
            raise HTTPException(status_code=503, detail="Database not initialized. Please wait for the service to start.")
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required for upload")
        
        supported_extensions = {'.txt', '.md', '.markdown', '.pdf', '.zip'}
        
        base_kb_folder = Path(os.getenv("KB_FOLDER", "KB"))
        # Use session-specific KB folder
        session_kb_folder = base_kb_folder / "sessions" / session_id
        
        # Clear existing session KB folder to replace KB
        if session_kb_folder.exists():
            shutil.rmtree(session_kb_folder)
        session_kb_folder.mkdir(parents=True, exist_ok=True)
        
        # Upload to session-specific KB folder
        upload_dir = session_kb_folder
        
        user_id = user_id or "anonymous"
        all_documents = []
        uploaded_files = []
        total_size = 0
        
        for file in files:
            if not file.filename:
                continue
            
            file_extension = Path(file.filename).suffix.lower()
            
            if file_extension == '.zip':
                with tempfile.TemporaryDirectory() as temp_dir:
                    zip_path = Path(temp_dir) / file.filename
                    with open(zip_path, "wb") as buffer:
                        shutil.copyfileobj(file.file, buffer)
                    
                    extract_dir = Path(temp_dir) / "extracted"
                    extract_dir.mkdir(exist_ok=True)
                    
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(extract_dir)
                    
                    for root, dirs, files_in_zip in os.walk(extract_dir):
                        for f in files_in_zip:
                            if Path(f).suffix.lower() in supported_extensions:
                                src_path = Path(root) / f
                                rel_path = src_path.relative_to(extract_dir)
                                dest_path = upload_dir / rel_path
                                dest_path.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(src_path, dest_path)
                                uploaded_files.append(str(rel_path))
                                total_size += dest_path.stat().st_size
                    
                    folder_docs = process_folder(upload_dir, str(kb_folder))
                    all_documents.extend(folder_docs)
            
            elif file_extension in supported_extensions:
                # Use original filename
                saved_filename = file.filename
                file_path = upload_dir / saved_filename
                
                # Handle filename conflicts
                counter = 1
                while file_path.exists():
                    name_part = Path(file.filename).stem
                    ext_part = Path(file.filename).suffix
                    saved_filename = f"{name_part}_{counter}{ext_part}"
                    file_path = upload_dir / saved_filename
                    counter += 1
                
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                total_size += file_path.stat().st_size
                
                try:
                    doc = process_uploaded_file(file_path, file.filename, str(kb_folder))
                    all_documents.append(doc)
                    uploaded_files.append(saved_filename)
                except Exception:
                    continue
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file format: {file_extension}. Supported formats: .txt, .md, .pdf, .zip"
                )
        
        if not all_documents:
            raise HTTPException(status_code=400, detail="No valid files were processed")
        
        # Rebuild index from scratch with new documents (session-specific KB fully replaced)
        engine = get_rag_engine(session_id=session_id)
        
        index_path = Path(engine.index_path)
        if index_path.exists():
            if index_path.is_dir():
                shutil.rmtree(index_path)
            else:
                index_path.unlink()
                pkl_path = Path(str(index_path) + ".pkl")
                if pkl_path.exists():
                    pkl_path.unlink()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=engine.chunk_size,
            chunk_overlap=engine.chunk_overlap,
            length_function=len
        )
        split_docs = text_splitter.split_documents(all_documents)
        engine.vectorstore = FAISS.from_documents(split_docs, engine.embedding_model)
        engine.vectorstore.save_local(engine.index_path)
        
        # Update session engine cache
        session_rag_engines[session_id] = engine
        
        kb_name = kb_name or f"{len(uploaded_files)} file(s)"
        kb_record = db.save_knowledge_base(
            user_id, 
            kb_name, 
            str(upload_dir),
            metadata={"files": uploaded_files, "file_count": len(uploaded_files), "session_id": session_id}
        )
        
        db.log_analytics("kb_upload", {
            "user_id": user_id,
            "session_id": session_id,
            "file_count": len(uploaded_files),
            "file_size": total_size,
            "files": uploaded_files
        })
        
        return {
            "message": f"Successfully uploaded {len(uploaded_files)} file(s) to your session.",
            "session_id": session_id,
            "files_uploaded": uploaded_files,
            "file_count": len(uploaded_files),
            "kb_record": {
                "kb_name": kb_record["kb_name"],
                "created_at": kb_record["created_at"].isoformat()
            }
        }
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Error uploading files")


@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    """Get analytics data"""
    try:
        if db is None:
            raise HTTPException(status_code=503, detail="Database not initialized. Please wait for the service to start.")
        
        all_sessions = list(db.sessions.find())
        total_sessions = len(all_sessions)
        
        all_messages = list(db.messages.find())
        total_messages = len(all_messages)
        
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        sessions_today = len(list(db.sessions.find({"created_at": {"$gte": today_start}})))
        messages_today = len(list(db.messages.find({"timestamp": {"$gte": today_start}})))
        
        user_messages = [msg["content"] for msg in all_messages if msg["role"] == "user"]
        query_counts = Counter(user_messages)
        popular_queries = [
            {"query": query, "count": count}
            for query, count in query_counts.most_common(10)
        ]
        
        return AnalyticsResponse(
            total_sessions=total_sessions,
            total_messages=total_messages,
            sessions_today=sessions_today,
            messages_today=messages_today,
            popular_queries=popular_queries
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Error getting analytics")


@app.get("/knowledge-bases")
async def list_knowledge_bases(user_id: Optional[str] = None):
    """List knowledge bases from database"""
    try:
        if db is None:
            raise HTTPException(status_code=503, detail="Database not initialized. Please wait for the service to start.")
        user_id = user_id or "anonymous"
        kbs = db.get_user_knowledge_bases(user_id)
        return {
            "user_id": user_id,
            "knowledge_bases": [
                {
                    "kb_name": kb["kb_name"],
                    "file_path": kb["file_path"],
                    "created_at": kb["created_at"].isoformat(),
                    "metadata": kb.get("metadata", {})
                }
                for kb in kbs
            ]
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Error listing knowledge bases")


@app.get("/knowledge-base/files")
async def list_kb_files(session_id: Optional[str] = None):
    """List all files in the knowledge base (session-specific if session_id provided)"""
    try:
        base_kb_folder = Path(os.getenv("KB_FOLDER", "KB"))
        
        # If session_id provided, use session-specific folder
        if session_id:
            kb_folder = base_kb_folder / "sessions" / session_id
            kb_folder.mkdir(parents=True, exist_ok=True)
            
            # Auto-seed session folder with default KB files if empty
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
            # Skip sessions subfolder when listing base KB
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
    """Get the content of a specific KB file (session-specific if session_id provided)"""
    try:
        base_kb_folder = Path(os.getenv("KB_FOLDER", "KB"))
        
        # If session_id provided, use session-specific folder
        if session_id:
            kb_folder = base_kb_folder / "sessions" / session_id
            kb_folder.mkdir(parents=True, exist_ok=True)
            
            # Auto-seed session folder with default KB files if empty
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
    """Delete a KB file (session-specific if session_id provided) and rebuild index"""
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id is required for delete")
            
        base_kb_folder = Path(os.getenv("KB_FOLDER", "KB"))
        kb_folder = base_kb_folder / "sessions" / session_id
        kb_folder.mkdir(parents=True, exist_ok=True)
        
        # Auto-seed session folder with default KB files if empty
        if not any(kb_folder.iterdir()):
            supported_extensions = {'.txt', '.md', '.markdown', '.pdf'}
            if base_kb_folder.exists():
                for item in base_kb_folder.iterdir():
                    if item.is_file() and item.suffix.lower() in supported_extensions:
                        shutil.copy2(item, kb_folder / item.name)
        
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
        
        # Delete the file
        file_full_path.unlink()
        
        # Rebuild index from remaining documents (session-specific)
        engine = get_rag_engine(session_id=session_id)
        
        from backend.file_processor import extract_text_from_file
        from langchain.docstore.document import Document
        documents = []
        
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
        
        # Update session engine cache
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

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


app = FastAPI(title="Advanced RAG Chatbot API", version="2.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:4173", "http://localhost:80"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database (will be set during startup)
db: Optional[Database] = None

# Initialize RAG engines (one per model type)
rag_engines: Dict[str, RAGEngine] = {}


@app.on_event("startup")
async def startup_event():
    """Initialize database connection and RAG engine on startup with retry logic"""
    global db
    print("\n" + "="*60)
    print("Starting RAG Chatbot Backend...")
    print("="*60)
    
    # Initialize database with retry logic
    print("\n[1/3] Connecting to MongoDB...")
    max_retries = 10  # Reduced from 30 to avoid long waits
    retry_delay = 2
    mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
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
                if attempt == 0:
                    print(f"  Connection failed: {error_msg}")
                    print(f"  Retrying MongoDB connection (will try {max_retries} times)...")
                elif attempt % 3 == 0:  # Show progress every 3 attempts
                    print(f"  Still retrying... (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(retry_delay)
            else:
                print(f"\n✗ Failed to connect to MongoDB after {max_retries} attempts.")
                print(f"  Error: {error_msg}")
                print("\n  To fix this, choose one option:")
                print("\n  Option 1: Install MongoDB locally")
                print("    macOS:")
                print("      brew tap mongodb/brew")
                print("      brew install mongodb-community")
                print("      brew services start mongodb-community")
                print("    Windows:")
                print("      choco install mongodb")
                print("      # Or download from: https://www.mongodb.com/try/download/community")
                print("      net start MongoDB")
                print("\n  Option 2: Use Docker:")
                print("    docker run -d -p 27017:27017 --name mongodb mongo:7")
                print("\n  Option 3: Use MongoDB Atlas (cloud):")
                print("    Sign up at https://www.mongodb.com/cloud/atlas")
                print("    Then set MONGODB_URI to your Atlas connection string")
                print(f"\n  Current MONGODB_URI: {mongodb_uri}")
                raise
    
    # Pre-initialize RAG engine to load models during startup (not on first request)
    print("\n[2/3] Initializing RAG engine...")
    print("  This may take 30-60 seconds when using local models...")
    try:
        engine = get_rag_engine()
        print("✓ RAG engine initialized successfully.")
    except Exception as e:
        print(f"⚠ Warning: Could not pre-initialize RAG engine: {e}")
        print("  RAG engine will be initialized on first request.")
    
    print("\n[3/3] Startup complete!")
    print("="*60)
    print("Backend is ready to accept requests.")
    print("API available at: http://0.0.0.0:8000")
    print("API docs available at: http://0.0.0.0:8000/docs")
    print("="*60 + "\n")


def get_rag_engine() -> RAGEngine:
    """Get or create RAG engine - auto-detects local model if available"""
    use_local_env = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
    openai_api_key = os.getenv("OPENAI_API_KEY")
    kb_folder = os.getenv("KB_FOLDER", "KB")
    
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
    
    engine_key = "local" if use_local else "openai"
    
    if engine_key not in rag_engines:
        if use_local and not local_model_exists:
            raise ValueError(
                f"Local model file not found at {model_path}. "
                "Please download the model or set USE_LOCAL_LLM=false to use OpenAI."
            )
        
        rag_engines[engine_key] = RAGEngine(
            kb_folder=kb_folder,
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
    """Detailed health check"""
    try:
        engine = get_rag_engine()
        local_available = os.path.exists("models/Llama-3.2-3B-Instruct-Q4_K_M.gguf")
        openai_available = bool(os.getenv("OPENAI_API_KEY"))
        use_local = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
        
        return HealthResponse(
            status="ok",
            message="RAG engine is ready",
            models_available={
                "openai": openai_available,
                "local": local_available,
                "active": "local" if use_local else "openai"
            }
        )
    except Exception:
        raise HTTPException(status_code=500, detail="RAG engine error")


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
        
        # Get RAG engine
        engine = get_rag_engine()
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
    """Upload knowledge base files"""
    try:
        if db is None:
            raise HTTPException(status_code=503, detail="Database not initialized. Please wait for the service to start.")
        if not files or len(files) == 0:
            raise HTTPException(status_code=400, detail="No files provided")
        
        supported_extensions = {'.txt', '.md', '.markdown', '.pdf', '.zip'}
        
        if session_id:
            upload_dir = Path("KB") / "sessions" / session_id
        else:
            session_id = str(uuid.uuid4())
            upload_dir = Path("KB") / "sessions" / session_id
        
        upload_dir.mkdir(parents=True, exist_ok=True)
        
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
                    
                    folder_docs = process_folder(upload_dir, str(upload_dir))
                    for doc in folder_docs:
                        doc.metadata["session_id"] = session_id
                        doc.metadata["is_session_upload"] = True
                    all_documents.extend(folder_docs)
            
            elif file_extension in supported_extensions:
                file_id = str(uuid.uuid4())
                saved_filename = f"{file_id}{file_extension}"
                file_path = upload_dir / saved_filename
                
                with open(file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                total_size += file_path.stat().st_size
                
                try:
                    doc = process_uploaded_file(file_path, file.filename, str(upload_dir))
                    doc.metadata["session_id"] = session_id
                    doc.metadata["is_session_upload"] = True
                    all_documents.append(doc)
                    uploaded_files.append(file.filename)
                except Exception:
                    continue
            else:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported file format: {file_extension}. Supported formats: .txt, .md, .pdf, .zip"
                )
        
        if not all_documents:
            raise HTTPException(status_code=400, detail="No valid files were processed")
        
        for engine in rag_engines.values():
            engine.add_documents_to_index(all_documents)
        
        kb_name = kb_name or f"{len(uploaded_files)} file(s)"
        kb_record = db.save_knowledge_base(
            user_id, 
            kb_name, 
            str(upload_dir),
            metadata={"files": uploaded_files, "file_count": len(uploaded_files)}
        )
        
        db.log_analytics("kb_upload", {
            "user_id": user_id,
            "file_count": len(uploaded_files),
            "file_size": total_size,
            "files": uploaded_files
        })
        
        return {
            "message": f"Successfully uploaded {len(uploaded_files)} file(s) to your session",
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
async def list_kb_files():
    """List all files in the knowledge base directory"""
    try:
        kb_folder = Path(os.getenv("KB_FOLDER", "KB"))
        supported_extensions = {'.txt', '.md', '.markdown', '.pdf'}
        files = []
        
        if not kb_folder.exists():
            return {"files": [], "total": 0}
        
        for root, dirs, filenames in os.walk(kb_folder):
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
            "total": len(files)
        }
    except Exception:
        raise HTTPException(status_code=500, detail="Error listing KB files")


@app.get("/knowledge-base/files/{file_path:path}")
async def get_kb_file_content(file_path: str):
    """Get the content of a specific KB file"""
    try:
        kb_folder = Path(os.getenv("KB_FOLDER", "KB"))
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

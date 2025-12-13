# RAG Knowledge Base Assistant with Multi-Turn Reasoning

A Retrieval-Augmented Generation (RAG) system to answer questions from a knowledge base with OpenAI integration, local Llama support for privacy, multi-turn reasoning, session tracking, analytics, and knowledge base management.

## Features

### Core Features
- **OpenAI & Llama Integration**: Uses OpenAI with Local Llama support for privacy-conscious users (see Local Llama Setup)
- **Multi-Turn Reasoning**: Context-aware conversations with chat history
- **Session Tracking**: MongoDB-based session management and analytics
- **Knowledge Base Management**: Upload, view, and browse KB files (supports .txt, .md, .pdf)
- **Agentic Capabilities**: Step-by-step reasoning and context-aware responses
- **Intent Classification**: Zero-shot intent detection for optimized responses
- **RAG with Reranking**: FAISS vector search with cross-encoder reranking


### Production Features
- **Dockerized**: Docker Compose setup for easy deployment
- **Analytics Dashboard**: Real-time usage statistics with popular queries
- **Session Management**: Persistent chat sessions with message history
- **FastAPI Backend**: High-performance async API
- **React UI**: Responsive chat interface with a clean layout

---
## Prerequisites

- Python 3.9+
- Node.js 18+
- Docker & Docker Compose (for containerized deployment)
- MongoDB (or use Docker Compose)
- OpenAI API key (for default OpenAI mode)

---

## Setup

### Option 1: Docker Deployment (Recommended)

1. **Clone the repository**
```bash
git clone https://github.com/RidhikaMadan/kb-rag.git
cd kb-rag
```

2. **Set environment variables**
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

3. **Start services**
```bash
docker compose up -d
```

4. **Access the application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

### Option 2: Local Development

1. **Create virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up MongoDB** (choose one option)

   **Option A: Install MongoDB locally**
   
   *macOS:*
   ```bash
   # Install Homebrew if you don't have it: https://brew.sh
   brew tap mongodb/brew
   brew install mongodb-community
   brew services start mongodb-community
   ```
   
   *Windows:*
   ```powershell
   # Option 1: Using Chocolatey (if installed)
   choco install mongodb
   
   # Option 2: Manual installation
   # 1. Download MongoDB Community Server from:
   #    https://www.mongodb.com/try/download/community
   # 2. Run the installer (.msi file)
   # 3. Choose "Complete" installation
   # 4. Install as a Windows Service (recommended)
   # 5. MongoDB will start automatically after installation
   
   # To start/stop MongoDB service manually:
   net start MongoDB
   # net stop MongoDB
   ```

   **Option B: Use Docker**
   ```bash
   docker run -d -p 27017:27017 --name mongodb mongo:7
   ```

   **Option C: Use MongoDB Atlas (cloud - free tier available)**
   - Sign up at https://www.mongodb.com/cloud/atlas
   - Create a free cluster
   - Get your connection string and set it as `MONGODB_URI`

4. **Set environment variables**
```bash
export OPENAI_API_KEY="your-api-key-here"
export MONGODB_URI="mongodb://localhost:27017/"
export MONGODB_DB="rag_chatbot"
```

5. **Regenerate embeddings (if updating KB files)**
```bash
# Delete old index to regenerate with updated KB content
rm -rf index.faiss/
# The index will be automatically regenerated on next startup
```

6. **Install frontend dependencies**
```bash
cd frontend
npm install
cd ..
```

7. **Start backend**
```bash
python -m uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

8. **Start frontend** (in another terminal)
```bash
cd frontend
npm run dev
```

9. **Open browser**
   - http://localhost:3000

---

## Local Llama Setup (For Privacy-Conscious Users)

If you're running the application locally and prefer not to use OpenAI (for privacy reasons), you can use the local Llama model instead. The provided download script uses 4-bit quantized Llama 3.2B. If you already have a Llama model file, place it in the `models/` directory.

### Setup Steps

Follow the local deployment instructions above with these additional steps before starting the backend. 

1. **Download the local model** (if not already done)
```bash
chmod +x download_model.sh
./download_model.sh
```

2. **Set environment variable to enable local Llama**
```bash
export USE_LOCAL_LLM="true"
```

3. **Start the backend** (no OpenAI API key needed)
```bash
python -m uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000
```

---

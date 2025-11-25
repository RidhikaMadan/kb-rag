# RAG Knowledge Base Assistant

A fully local and private Retrieval-Augmented Generation (RAG) system using 4-bit quantized LLaMA, semantic chunking, FAISS vector search, cross-encoder reranking, and zero-shot intent classification built with LangChain and HuggingFace to answer queries from a structured internal company knowledge base.

---

## Setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the model (Default: LLaMA 3.1 8B Instruct (q4_k_m))


```bash
chmod +x download_model.sh
./download_model.sh
```

### 4. Run app

```bash
python main.py
```


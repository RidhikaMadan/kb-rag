#!/bin/bash

# Start the FastAPI backend server
echo "Starting FastAPI backend server..."

# Set UTF-8 encoding to prevent Unicode errors
export PYTHONIOENCODING=utf-8
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8

# Note: OpenAI API key should be set via environment variable
# For local Llama usage (privacy), set USE_LOCAL_LLM=true instead

# Set MongoDB URI if not set
if [ -z "$MONGODB_URI" ]; then
    export MONGODB_URI="mongodb://localhost:27017/"
fi

if [ -z "$MONGODB_DB" ]; then
    export MONGODB_DB="rag_chatbot"
fi

python -m uvicorn backend.api:app --reload --host 0.0.0.0 --port 8000



#!/bin/bash

# Script to regenerate FAISS index with updated KB content
echo "Regenerating FAISS index with updated KB content..."
echo "This will delete the old index and create a new one from KB files."

# Delete old index
if [ -d "index.faiss" ]; then
    echo "Deleting old index..."
    rm -rf index.faiss/
    echo "Old index deleted."
else
    echo "No existing index found."
fi

echo ""
echo "The index will be automatically regenerated when you start the backend."
echo "Start the backend with: ./start_backend.sh or python -m uvicorn backend.api:app --reload"


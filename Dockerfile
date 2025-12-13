# Multi-stage build 

# Stage 1: Backend
FROM python:3.9-slim as backend

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/
COPY main.py .

# Expose backend port
EXPOSE 8000

# Stage 2: Frontend
FROM node:18-alpine as frontend

WORKDIR /app

# Copy frontend files
COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./

# Build frontend
RUN npm run build

# Stage 3: Final image
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies from backend stage
COPY --from=backend /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=backend /usr/local/bin /usr/local/bin

# Copy backend code
COPY backend/ ./backend/
COPY main.py .
COPY requirements.txt .

# Copy built frontend
COPY --from=frontend /app/dist ./frontend/dist

# Create directories for KB and models
RUN mkdir -p KB models index.faiss

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONIOENCODING=utf-8
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV KB_FOLDER=KB

# Expose ports
EXPOSE 8000

# Run backend server
CMD ["uvicorn", "backend.api:app", "--host", "0.0.0.0", "--port", "8000"]


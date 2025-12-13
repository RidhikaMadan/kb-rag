#!/bin/bash

# Rebuild and restart Docker containers
echo "Stopping containers..."
docker compose down

echo "Rebuilding containers..."
docker compose build --no-cache

echo "Starting containers..."
docker compose up -d

echo "Waiting for services to start..."
sleep 5

echo "Checking container status..."
docker compose ps

echo ""
echo "Services should be running:"
echo "  - Frontend: http://localhost:3000"
echo "  - Backend API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo ""
echo "View logs with: docker compose logs -f"


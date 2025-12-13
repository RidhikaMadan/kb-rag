# Quick Start Guide

### Using Docker (Recommended)

1. **Set your OpenAI API key:**
```bash
export OPENAI_API_KEY="your-api-key-here"
```

2. **Start docker:**
```bash
docker compose up -d --build
```

   Or, to rebuild from scratch:
```bash
./rebuild.sh
```

3. **Access the app:**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Local Development

1. **Start MongoDB** (choose one option):

   **Option A: Install locally**
   
   *macOS:*
   ```bash
   brew tap mongodb/brew
   brew install mongodb-community
   brew services start mongodb-community
   ```
   
   *Windows:*
   ```powershell
   # Using Chocolatey (if installed)
   choco install mongodb
   
   # Or download installer from:
   # https://www.mongodb.com/try/download/community
   # Run installer and choose "Install as Windows Service"
   
   # To start MongoDB service:
   net start MongoDB
   ```

   **Option B: Use Docker:**
   ```bash
   docker run -d -p 27017:27017 --name mongodb mongo:7
   ```

   **Option C: Use MongoDB Atlas (cloud):**
   - Sign up at https://www.mongodb.com/cloud/atlas
   - Create a free cluster and get connection string

2. **Set environment variables:**
```bash
export OPENAI_API_KEY="your-api-key-here"
export MONGODB_URI="mongodb://localhost:27017/"
```

3. **Start backend:**
```bash
chmod +x start_backend.sh
./start_backend.sh
```

4. **Start frontend (new terminal):**
```bash
chmod +x start_frontend.sh
./start_frontend.sh
```

5. **Open browser:**
   - http://localhost:3000


## Troubleshooting

**Frontend shows 403 or redirect errors:**
- Rebuild the containers: `./rebuild.sh` or `docker compose up -d --build`
- Ensure the frontend was built properly in the Docker image
- Check logs: `docker compose logs frontend`

**Backend won't start:**
- Check MongoDB is running: `docker compose ps`
- Verify OPENAI_API_KEY is set
- Check port 8000 is available
- Check logs: `docker compose logs backend`

**Frontend won't connect to backend:**
- Ensure backend is running: `docker compose ps`
- Check browser console for errors
- Verify nginx is proxying correctly: `docker compose logs frontend`
- Try accessing backend directly: http://localhost:8000/health

**Model toggle not working:**
- For local model, ensure model file exists in `models/` directory
- Check `models/` directory has the `.gguf` file
- Verify volume mount in docker-compose.yml


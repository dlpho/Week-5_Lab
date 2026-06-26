Ho, Denise Liana P. 
# Student Handbook Chatbot

RAG-powered school policy chatbot for Week 5.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
- [Ollama](https://ollama.com) installed and running

```bash
ollama pull gemma3:1b
ollama pull nomic-embed-text
```

### Run

```bash
docker compose up --build
```

| | URL |
|---|---|
| Frontend (Chat UI) | http://localhost:8501 |
| Backend (API) | http://localhost:8000/docs |

### Stop

```bash
# Keep data
docker compose down

# Wipe data and start fresh
docker compose down -v
```

### Project Structure

```
├── core/
│   ├── engine.py       # RAG + Memory + Guardrails (Week 3-4)
│   └── llmops.py       # MLFlow tracing + JSON logging
├── api.py              # FastAPI /chat/stream SSE endpoint
├── app.py              # Streamlit chat UI
├── school_handbook.pdf # Knowledge base
├── Dockerfile
├── docker-compose.yml
└── supervisord.conf    # Runs both servers in one container
```
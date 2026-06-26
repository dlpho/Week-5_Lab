# Student Handbook Chatbot
Ho, Denise Liana P. 

RAG-powered school policy chatbot for Week 5.

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) running
- [Ollama](https://ollama.com) installed and running

Pull the required models:
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
| Frontend (Chat UI) | http://localhost:7860|
| Backend (API) | http://localhost:8000/docs |
| LLMOps (MLFlow) | http://localhost:5000 |


### LLMOps & Tracing (Week 5 Requirement)
This project logs JSON latency/token metrics to the backend console and captures full trace trees in MLFlow. To view the traces, open a new terminal in the project root and run:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db

Then navigate to http://localhost:5000 to inspect the rag_retrieval and llm_inference spans.

### Stop

```bash
# Keep data
docker compose down

# Wipe data and start fresh
docker compose down -v
```



### Project Structure

```
week-5-lab/
│
├── docker-compose.yml
│
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── api.py
│   ├── school_handbook.pdf
│   └── core/ 
│       ├── engine.py
│       └── llmops.py
│
└── frontend/
    ├── Dockerfile
    ├── requirements.txt
    └── app.py
```
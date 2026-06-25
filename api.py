"""
FastAPI backend — exposes POST /chat/stream as a Server-Sent Events endpoint.

Architecture (from slide deck):
    User → Gradio/Streamlit UI → FastAPI :8000 → RAG core → LLM

Both the Streamlit frontend and any external API consumer call this endpoint;
the RAG + Memory + Guardrails logic lives entirely in core/engine.py.
"""

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from core.engine import generate_chat_stream

app = FastAPI(title="School Handbook Support API", version="1.0.0")

# Allow requests from the Streamlit frontend (port 8501) and any origin
# during development.  Tighten this in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------
# Request / response schemas
# --------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str


# --------------------------------------------------------------------------
# Health check — used by Docker HEALTHCHECK and load balancers
# --------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


# --------------------------------------------------------------------------
# POST /chat/stream — SSE endpoint
#
# SSE format (per HTML spec):
#   data: <payload>\n\n
#
# Newlines inside the payload are escaped to \\n so they don't break the
# framing.  The Streamlit client reverses this when it renders.
# --------------------------------------------------------------------------
@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="'message' must not be empty.")

    async def event_generator():
        """
        generate_chat_stream is a synchronous generator (it uses LangChain's
        .stream() which is sync).  We run it in a thread-pool executor so we
        don't block the FastAPI event loop.
        """
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[str | None] = asyncio.Queue()

        def producer():
            try:
                for chunk in generate_chat_stream(req.message):
                    # put_nowait is safe from a worker thread
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            finally:
                # Sentinel value signals end-of-stream
                loop.call_soon_threadsafe(queue.put_nowait, None)

        # Run the blocking generator in the default thread pool
        loop.run_in_executor(None, producer)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            # Escape newlines to keep SSE framing intact
            encoded = chunk.replace("\n", "\\n")
            yield f"data: {encoded}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
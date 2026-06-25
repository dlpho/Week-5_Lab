from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from core.engine import generate_chat_stream

app = FastAPI(title="Dual Channel Support API")

class ChatRequest(BaseModel):
    message: str

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    if not req.message:
        raise HTTPException(status_code=400, detail="Message is required")
        
    def event_stream():
        for chunk in generate_chat_stream(req.message):
            yield f"data: {chunk}\n\n"
            
    return StreamingResponse(event_stream(), media_type="text/event-stream")
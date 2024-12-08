import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import json

from models.model_manager import ModelManager
from routing.query_router import QueryRouter
from models.chat_session import ChatSession

# Global instances
model_manager: Optional[ModelManager] = None
query_router: Optional[QueryRouter] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    global model_manager, query_router
    print("Initializing services...")
    model_manager = ModelManager(debug=False)
    query_router = QueryRouter(debug=False)
    
    # Pre-load all models
    print("Loading models...")
    for group in model_manager.available_groups():
        print(f"Loading {group} model...")
        model_manager.load_model(group)
        
    yield
    
    # Cleanup (if needed)
    print("Shutting down...")

app = FastAPI(title="LlamaChat API", lifespan=lifespan)

# Pydantic models for request/response validation
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

async def generate_streaming_response(chat_session, prompt, max_tokens, temperature):
    """Generate streaming response chunks."""
    try:
        # First, trigger model selection by starting response generation
        # This will select the model group based on the prompt
        response_generator = chat_session.generate_response(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )
        
        # Now we can safely send the model group
        yield json.dumps({
            "type": "meta",
            "model_group": chat_session.model_group
        }) + "\n"
        
        # Stream the response
        for chunk in response_generator:
            yield json.dumps({
                "type": "content",
                "text": chunk
            }) + "\n"
            
            # Add a small delay to ensure chunks are processed separately
            await asyncio.sleep(0.01)
            
    except Exception as e:
        yield json.dumps({
            "type": "error",
            "detail": str(e)
        }) + "\n"

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat responses chunk by chunk."""
    if not model_manager or not query_router:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    # Create a new chat session
    chat_session = ChatSession(model_manager, debug=True)  # Enable debug mode
    
    # Add historical messages to the session
    for msg in request.messages[:-1]:
        if msg.role in ["user", "assistant"]:
            chat_session.conversation_history.append({
                "user" if msg.role == "user" else "assistant": msg.content
            })
    
    # Get the last message (current query)
    current_query = next(
        msg.content for msg in reversed(request.messages)
        if msg.role == "user"
    )
    
    return StreamingResponse(
        generate_streaming_response(
            chat_session,
            current_query,
            request.max_tokens,
            request.temperature
        ),
        media_type="text/event-stream"
    )

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server."""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()
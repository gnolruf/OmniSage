import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, field_serializer
from typing import List, Optional, AsyncGenerator
import uvicorn
import json

from models.model_manager import ModelManager
from routing.query_router import QueryRouter
from models.chat_session import ChatSession
from database_manager import DatabaseManager

# Global instances
model_manager: Optional[ModelManager] = None
query_router: Optional[QueryRouter] = None
db_manager: Optional[DatabaseManager] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    global model_manager, query_router, db_manager
    print("Initializing services...")
    model_manager = ModelManager(debug=False)
    query_router = QueryRouter(debug=False)
    db_manager = DatabaseManager("postgresql://postgres:23441873@localhost:5432/llamachat")
    
    # Pre-load all models
    print("Loading models...")
    for group in model_manager.available_groups():
        print(f"Loading {group} model...")
        model_manager.load_model(group)
        
    yield
    
    # Cleanup
    print("Shutting down...")
    if db_manager:
        db_manager.close()

app = FastAPI(title="LlamaChat API", lifespan=lifespan)

# Pydantic models for request/response validation
class Message(BaseModel):
    role: str
    content: str

class ChatMetadata(BaseModel):
    title: str

class Chat(BaseModel):
    id: int
    title: str
    created_at: datetime
    updated_at: datetime
    latest_message: Optional[str] = None

    @field_serializer('created_at', 'updated_at')
    def serialize_datetime(self, dt: datetime) -> str:
        return dt.isoformat()

class ChatRequest(BaseModel):
    messages: List[Message]
    chat_id: Optional[int] = None
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7

async def generate_streaming_response(chat_session, prompt, chat_id, max_tokens, temperature) -> AsyncGenerator[str, None]:
    """Generate streaming response chunks."""
    try:
        # Generate response using synchronous method
        response_gen = chat_session.generate_response(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )
        
        # First, yield the model group information
        yield json.dumps({
            "type": "meta",
            "model_group": chat_session.model_group
        }) + "\n"
        
        # Stream the response chunks
        for chunk in response_gen:
            yield json.dumps({
                "type": "content",
                "text": chunk
            }) + "\n"
            await asyncio.sleep(0.01)  # Small delay for proper streaming
            
    except Exception as e:
        yield json.dumps({
            "type": "error",
            "detail": str(e)
        }) + "\n"

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat responses chunk by chunk."""
    if not model_manager or not query_router or not db_manager:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    # Create a new chat session
    chat_session = ChatSession(
        model_manager,
        db_manager,
        chat_id=request.chat_id,
        debug=True
    )
    
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
    
    # Save user message to database if we have a chat_id
    if request.chat_id:
        db_manager.save_message(request.chat_id, "user", current_query)
    
    return StreamingResponse(
        generate_streaming_response(
            chat_session,
            current_query,
            request.chat_id,
            request.max_tokens,
            request.temperature
        ),
        media_type="text/event-stream"
    )

@app.post("/chats", response_model=Chat)
async def create_chat(chat: ChatMetadata):
    """Create a new chat session."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    chat_id = db_manager.create_chat(chat.title)
    chats = db_manager.get_chats()
    return Chat(
        id=chats[0]["id"],
        title=chats[0]["title"],
        created_at=chats[0]["created_at"],
        updated_at=chats[0]["updated_at"],
        latest_message=chats[0]["latest_message"]
    )

@app.get("/chats", response_model=List[Chat])
async def list_chats():
    """Get all chat sessions."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    chats = db_manager.get_chats()
    # Convert the database rows to Chat models
    return [
        Chat(
            id=chat["id"],
            title=chat["title"],
            created_at=chat["created_at"],
            updated_at=chat["updated_at"],
            latest_message=chat["latest_message"]
        )
        for chat in chats
    ]

@app.get("/chats/{chat_id}/messages")
async def get_chat_messages(chat_id: int):
    """Get all messages for a specific chat."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    messages = db_manager.get_chat_messages(chat_id)
    # Ensure all datetime fields are converted to strings
    for msg in messages:
        msg["created_at"] = msg["created_at"].isoformat()
    return messages

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: int):
    """Delete a chat session."""
    if not db_manager:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    db_manager.delete_chat(chat_id)
    return {"status": "success", "message": f"Chat {chat_id} deleted"}

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server."""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()

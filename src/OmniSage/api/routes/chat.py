from fastapi import APIRouter, HTTPException, Request
from typing import List
from fastapi.responses import StreamingResponse

from src.omnisage.api.models.schemas import (
    Message,
    ChatMetadata,
    Chat,
    ChatRequest
)
from src.omnisage.api.utils import generate_streaming_response
from src.omnisage.core.chat import ChatSession
from src.omnisage.database.manager import DatabaseManager
from src.omnisage.database.config import DatabaseConfig

router = APIRouter(prefix="/chat", tags=["chat"])

def get_db_manager():
    """Get database manager instance with proper connection string."""
    db_config = DatabaseConfig()
    return DatabaseManager(db_config.get_connection_string())

@router.post("/stream")
async def chat_stream(request: Request, chat_request: ChatRequest):
    """Stream chat responses chunk by chunk."""
    try:
        # Get the model_manager from app state
        model_manager = request.app.state.model_manager
        if not model_manager:
            raise HTTPException(
                status_code=500,
                detail="Model manager not initialized"
            )
            
        db_manager = get_db_manager()
        
        # Create chat session
        chat_session = ChatSession(
            model_manager,
            db_manager,
            chat_id=chat_request.chat_id,
            debug=True
        )
        
        # Add historical messages to session
        for msg in chat_request.messages[:-1]:
            if msg.role in ["user", "assistant"]:
                chat_session.conversation_history.append({
                    "user" if msg.role == "user" else "assistant": msg.content
                })
        
        # Get current query
        current_query = next(
            msg.content for msg in reversed(chat_request.messages)
            if msg.role == "user"
        )
        
        # Save user message if we have a chat_id
        if chat_request.chat_id:
            db_manager.save_message(chat_request.chat_id, "user", current_query)
        
        # Generate streaming response
        return StreamingResponse(
            generate_streaming_response(
                chat_session,
                current_query,
                chat_request.chat_id,
                chat_request.max_tokens,
                chat_request.temperature
            ),
            media_type="text/event-stream"
        )
    except Exception as e:
        print(f"Chat error: {str(e)}")  # Add error logging
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chats", response_model=Chat)
async def create_chat(chat: ChatMetadata):
    """Create a new chat session."""
    try:
        db_manager = get_db_manager()
        chat_id = db_manager.create_chat(chat.title)
        chats = db_manager.get_chats()
        return Chat(
            id=chats[0]["id"],
            title=chats[0]["title"],
            created_at=chats[0]["created_at"],
            updated_at=chats[0]["updated_at"],
            latest_message=chats[0]["latest_message"]
        )
    except Exception as e:
        print(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chats", response_model=List[Chat])
async def list_chats():
    """Get all chat sessions."""
    try:
        db_manager = get_db_manager()  # Use the helper function
        chats = db_manager.get_chats()
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
    except Exception as e:
        print(f"Database error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Database error: {str(e)}"
        )

@router.get("/chats/{chat_id}/messages", response_model=List[Message])
async def get_chat_messages(chat_id: int):
    """Get all messages for a specific chat."""
    try:
        db_manager = get_db_manager()  # Use helper function
        messages = db_manager.get_chat_messages(chat_id)
        return messages
    except Exception as e:
        print(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: int):
    """Delete a chat session."""
    try:
        db_manager = get_db_manager()  # Use helper function
        db_manager.delete_chat(chat_id)
        return {"status": "success", "message": f"Chat {chat_id} deleted"}
    except Exception as e:
        print(f"Database error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
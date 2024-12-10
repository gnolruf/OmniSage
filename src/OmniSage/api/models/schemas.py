from pydantic import BaseModel, Field, field_serializer
from typing import List, Optional
from datetime import datetime

class Message(BaseModel):
    """Schema for chat messages."""
    role: str = Field(..., description="Role of the message sender (user or assistant)")
    content: str = Field(..., description="Content of the message")
    model_group: Optional[str] = Field(None, description="Model group used for assistant responses")
    created_at: Optional[datetime] = Field(None, description="Timestamp of message creation")

    @field_serializer('created_at')
    def serialize_datetime(self, dt: datetime | None) -> str | None:
        return dt.isoformat() if dt else None

    class Config:
        json_schema_extra = {
            "example": {
                "role": "user",
                "content": "Hello, how can you help me today?",
                "model_group": None,
                "created_at": "2024-12-08T10:00:00"
            }
        }

class ChatMetadata(BaseModel):
    """Schema for chat creation."""
    title: str = Field(..., description="Title of the chat session")

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Python Programming Help"
            }
        }

class Chat(BaseModel):
    """Schema for chat sessions."""
    id: int = Field(..., description="Unique identifier for the chat")
    title: str = Field(..., description="Title of the chat session")
    created_at: datetime = Field(..., description="Timestamp of chat creation")
    updated_at: datetime = Field(..., description="Timestamp of last update")
    latest_message: Optional[str] = Field(None, description="Content of the most recent message")

    @field_serializer('created_at', 'updated_at')
    def serialize_datetime(self, dt: datetime) -> str:
        return dt.isoformat()

    class Config:
        json_schema_extra = {
            "example": {
                "id": 1,
                "title": "Python Programming Help",
                "created_at": "2024-12-08T10:00:00",
                "updated_at": "2024-12-08T10:05:00",
                "latest_message": "How can I help you with Python today?"
            }
        }

class ChatRequest(BaseModel):
    """Schema for chat completion requests."""
    messages: List[Message] = Field(..., description="List of conversation messages")
    chat_id: Optional[int] = Field(None, description="ID of the chat session")
    max_tokens: Optional[int] = None  # Remove default value
    temperature: Optional[float] = Field(0.7, description="Temperature for response generation")

    class Config:
        json_schema_extra = {
            "example": {
                "messages": [
                    {"role": "user", "content": "Hello, how can you help me today?"}
                ],
                "chat_id": 1,
                "max_tokens": None,
                "temperature": 0.7
            }
        }

class ChatResponse(BaseModel):
    """Schema for chat completion responses."""
    model_group: str = Field(..., description="Model group used for response")
    content: str = Field(..., description="Generated response content")

    class Config:
        json_schema_extra = {
            "example": {
                "model_group": "default",
                "content": "Hello! I'm an AI assistant. I can help you with various tasks..."
            }
        }

class StreamingChunk(BaseModel):
    """Schema for streaming response chunks."""
    type: str = Field(..., description="Type of chunk (meta, content, or error)")
    text: Optional[str] = Field(None, description="Content text for content chunks")
    model_group: Optional[str] = Field(None, description="Model group for meta chunks")
    detail: Optional[str] = Field(None, description="Error detail for error chunks")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "content",
                "text": "Hello",
                "model_group": None,
                "detail": None
            }
        }
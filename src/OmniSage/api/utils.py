import asyncio
import json
from typing import AsyncGenerator

from src.omnisage.core.chat import ChatSession

async def generate_streaming_response(
    chat_session: ChatSession,
    prompt: str,
    chat_id: int | None,
    max_tokens: int,
    temperature: float
) -> AsyncGenerator[str, None]:
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
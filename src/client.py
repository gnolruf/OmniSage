import requests
import json
from typing import List, Optional, Generator, Union, Dict
from dataclasses import dataclass

@dataclass
class Message:
    role: str
    content: str

class LlamaChatClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')

    def create_chat(self, title: str) -> dict:
        """Create a new chat session."""
        response = requests.post(
            f"{self.base_url}/chats",
            json={"title": title}
        )
        response.raise_for_status()
        return response.json()
    
    def list_chats(self) -> list:
        """Get list of all chats."""
        response = requests.get(f"{self.base_url}/chats")
        response.raise_for_status()
        return response.json()
    
    def get_chat_messages(self, chat_id: int) -> list:
        """Get all messages for a specific chat."""
        response = requests.get(f"{self.base_url}/chats/{chat_id}/messages")
        response.raise_for_status()
        return response.json()
    
    def delete_chat(self, chat_id: int):
        """Delete a chat session."""
        response = requests.delete(f"{self.base_url}/chats/{chat_id}")
        response.raise_for_status()
        return response.json()
    
    def chat_stream(
        self,
        messages: List[Message],
        chat_id: Optional[int] = None,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> Generator[Union[str, Dict], None, None]:
        """
        Stream a chat response from the server.
        
        Args:
            messages: List of conversation messages
            chat_id: Optional ID of the current chat session
            max_tokens: Maximum tokens to generate
            temperature: Temperature for response generation
            
        Yields:
            Either string chunks of the response or metadata dictionaries
        """
        url = f"{self.base_url}/chat/stream"
        
        data = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "chat_id": chat_id,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        with requests.post(url, json=data, stream=True) as response:
            response.raise_for_status()
            
            # Buffer for incomplete lines
            buffer = ""
            
            # Iterate over the raw bytes from the response
            for chunk in response.iter_content(chunk_size=1):
                if chunk:
                    # Decode the byte to string
                    char = chunk.decode('utf-8')
                    buffer += char
                    
                    # If we have a complete line
                    if char == '\n' and buffer.strip():
                        try:
                            data = json.loads(buffer.strip())
                            if data["type"] == "content":
                                yield data["text"]
                            elif data["type"] == "meta":
                                yield {"model_group": data["model_group"]}
                            elif data["type"] == "error":
                                raise RuntimeError(data["detail"])
                        except json.JSONDecodeError:
                            # Handle incomplete JSON
                            pass
                        buffer = ""
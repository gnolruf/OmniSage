import argparse
from typing import Optional
import requests
from datetime import datetime
from src.omnisage.core.client import LlamaChatClient, Message

def format_chat_list(chats: list[dict]) -> str:
    """Format chat list for display."""
    if not chats:
        return "No saved chats found."
        
    result = "\nSaved Chats:\n"
    for chat in chats:
        latest = chat["latest_message"] or ""
        if latest and len(latest) > 50:
            latest = latest[:47] + "..."
            
        # Parse the datetime string
        updated = datetime.fromisoformat(chat["updated_at"])
        result += f"{chat['id']}: {chat['title']} (Last updated: {updated.strftime('%Y-%m-%d %H:%M')})\n"
        result += f"    Latest: {latest}\n"
    
    return result

def chat_session(
    client: LlamaChatClient,
    chat_id: Optional[int] = None,
    debug: bool = False
):
    """Run an interactive chat session using the API client."""
    print("\nWelcome to LlamaChat!")
    print("Available commands:")
    print("  /new <title> - Create new chat")
    print("  /load - List and load saved chats")
    print("  /delete - Delete a chat")
    print("  /clear - Clear current conversation")
    print("  /quit - Exit the program")
    
    current_chat_id = chat_id
    conversation_history: list[Message] = []
    
    if current_chat_id:
        # Load existing chat
        messages = client.get_chat_messages(current_chat_id)
        conversation_history = [
            Message(role=msg["role"], content=msg["content"])
            for msg in messages
        ]
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue
            
        # Handle commands
        if user_input.startswith("/"):
            cmd_parts = user_input[1:].split(maxsplit=1)
            command = cmd_parts[0].lower()
            
            if command in ["quit", "exit"]:
                print("\nGoodbye!")
                break
                
            elif command == "new":
                if len(cmd_parts) < 2:
                    print("Please provide a title for the new chat")
                    continue
                    
                title = cmd_parts[1]
                chat = client.create_chat(title)
                current_chat_id = chat["id"]
                conversation_history = []
                print(f"\nCreated new chat: {title}")
                continue
                
            elif command == "load":
                chats = client.list_chats()
                print(format_chat_list(chats))
                
                chat_choice = input("\nEnter chat ID to load (or press Enter to cancel): ").strip()
                if not chat_choice:
                    continue
                    
                try:
                    chat_id = int(chat_choice)
                    messages = client.get_chat_messages(chat_id)
                    conversation_history = [
                        Message(role=msg["role"], content=msg["content"])
                        for msg in messages
                    ]
                    current_chat_id = chat_id
                    print(f"\nLoaded chat {chat_id}")
                except (ValueError, TypeError):
                    print("Invalid chat ID")
                except requests.exceptions.HTTPError:
                    print("Chat not found")
                continue
                
            elif command == "delete":
                chats = client.list_chats()
                print(format_chat_list(chats))
                
                chat_choice = input("\nEnter chat ID to delete (or press Enter to cancel): ").strip()
                if not chat_choice:
                    continue
                    
                try:
                    chat_id = int(chat_choice)
                    client.delete_chat(chat_id)
                    if chat_id == current_chat_id:
                        current_chat_id = None
                        conversation_history = []
                    print(f"\nDeleted chat {chat_id}")
                except (ValueError, TypeError):
                    print("Invalid chat ID")
                except requests.exceptions.HTTPError:
                    print("Chat not found")
                continue
                
            elif command == "clear":
                current_chat_id = None
                conversation_history = []
                print("\nConversation history cleared.")
                continue
        
        # Ensure we have an active chat
        if not current_chat_id:
            print("\nNo active chat. Please create a new chat first with /new <title>")
            continue
        
        try:
            conversation_history.append(Message(role="user", content=user_input))
            
            print("\nAssistant: ", end="", flush=True)
            
            full_response = []
            
            for chunk in client.chat_stream(
                conversation_history,
                chat_id=current_chat_id,  # Pass the current chat ID
                max_tokens=512,
                temperature=0.7
            ):
                if isinstance(chunk, dict):
                    if debug:
                        print(f"\nUsing model group: {chunk['model_group']}")
                else:
                    print(chunk, end="", flush=True)
                    full_response.append(chunk)
            
            # Add assistant's response to history
            conversation_history.append(
                Message(role="assistant", content="".join(full_response).strip())
            )
            print()
            
        except Exception as e:
            print(f"\nError: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Start a chat interface with Llama models")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--host", type=str, default="http://localhost:8000",
                       help="API server host URL")
    
    args = parser.parse_args()
    
    try:
        # Initialize only the client
        client = LlamaChatClient(base_url=args.host)
        
        # Start chat session without db_manager
        chat_session(client, debug=args.debug)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

import argparse
from typing import Optional
import requests
from client import LlamaChatClient, Message

def list_available_models(client: LlamaChatClient):
    """Print available models from the API server."""
    try:
        response = requests.get(f"{client.base_url}/models")
        response.raise_for_status()
        models = response.json()
        
        print("\nAvailable model groups:")
        for model in models:
            status = "loaded" if model["loaded"] else "not loaded"
            print(f"  - {model['name']} ({status})")
        print()
    except Exception as e:
        print(f"Error fetching models: {str(e)}")

def chat_session(client: LlamaChatClient, debug: bool = False):
    """Run an interactive chat session using the API client."""
    print("\nWelcome to LlamaChat!")
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to clear the conversation history\n")
    
    conversation_history: list[Message] = []
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit']:
            print("\nGoodbye!")
            break
        elif user_input.lower() == 'clear':
            conversation_history = []
            print("\nConversation history cleared.")
            continue
        elif not user_input:
            continue
        
        try:
            conversation_history.append(Message(role="user", content=user_input))
            
            print("\nAssistant: ", end="", flush=True)
            
            full_response = []
            word_buffer = []
            
            for chunk in client.chat_stream(conversation_history):
                if isinstance(chunk, dict):
                    if debug:
                        print(f"\nUsing model group: {chunk['model_group']}")
                else:
                    print(chunk, end="", flush=True)
                    full_response.append(chunk)
                    word_buffer.append(chunk)
                    
                    # Flush word buffer on spaces or punctuation
                    if chunk in " .,!?;:()\n":
                        word_buffer = []
            
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
    parser.add_argument("--list-models", action="store_true",
                       help="List available model groups and exit")
    parser.add_argument("--host", type=str, default="http://localhost:8000",
                       help="API server host URL")
    
    args = parser.parse_args()
    
    try:
        # Initialize client
        client = LlamaChatClient(base_url=args.host)
        
        # If --list-models flag is used, show available models and exit
        if args.list_models:
            list_available_models(client)
            return
        
        # Start chat session
        chat_session(client, debug=args.debug)
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

import argparse
from models.model_manager import ModelManager
from models.chat_session import ChatSession

def list_available_models(model_manager: ModelManager):
    """Print available model groups from config."""
    print("\nAvailable model groups:")
    for group in model_manager.available_groups():
        print(f"  - {group}")
    print()

def main():
    parser = argparse.ArgumentParser(description="Start a chat interface with Llama models")
    parser.add_argument("--n-threads", type=int,
                       help="Number of threads to use")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--list-models", action="store_true",
                       help="List available model groups and exit")
    
    args = parser.parse_args()
    
    try:
        # Initialize model manager
        model_manager = ModelManager(n_threads=args.n_threads, debug=args.debug)
        
        # If --list-models flag is used, show available models and exit
        if args.list_models:
            list_available_models(model_manager)
            return
        
        # Load all available models since we don't know which will be needed
        print("\nLoading models...")
        for group in model_manager.available_groups():
            print(f"Loading {group} model...")
            model_manager.load_model(group)
        
        # Create and start chat session
        chat = ChatSession(model_manager, debug=args.debug)
        chat.chat()
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
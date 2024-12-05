import argparse
from .models.llama_chat import LlamaChat

def main():
    parser = argparse.ArgumentParser(description="Start a chat interface with multiple Llama models")
    parser.add_argument("--n-threads", type=int, help="Number of threads to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
    
    args = parser.parse_args()
    
    try:
        chat = LlamaChat(
            n_threads=args.n_threads,
            debug=args.debug
        )
        chat.chat()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
import os
from llama_cpp import Llama
import argparse
from typing import Optional, List
import textwrap
import shutil
from pathlib import Path
from huggingface_hub import HfApi
import re

class ModelManager:
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory to store downloaded models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Check if model exists in local storage."""
        model_file = self.models_dir / model_name
        return model_file if model_file.exists() else None
    
    def save_model(self, source_path: str, model_name: str):
        """Save a model to the models directory."""
        target_path = self.models_dir / model_name
        shutil.copy2(source_path, target_path)
        return target_path

    @staticmethod
    def list_available_models(repo_id: str) -> List[str]:
        """List available model files in the HuggingFace repository."""
        api = HfApi()
        files = api.list_repo_files(repo_id)
        return [f for f in files if f.endswith('.gguf')]

class LlamaChat:
    def __init__(self, 
                 base_model_name: str,
                 quantization: str,
                 repo_id: Optional[str] = None,
                 n_ctx: int = 2048,
                 n_threads: Optional[int] = None):
        """
        Initialize the LlamaChat interface.
        
        Args:
            base_model_name: Base name of the model
            quantization: Quantization format (e.g., 'Q4_0', 'Q5_K_M')
            repo_id: HuggingFace repo ID for downloading the model
            n_ctx: Context window size
            n_threads: Number of threads to use for inference
        """
        self.model_manager = ModelManager()
        self.model_path = self._get_or_download_model(base_model_name, quantization, repo_id)
        
        self.llm = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_threads=n_threads or os.cpu_count()
        )
        
        self.conversation_history = []
        self.wrapper = textwrap.TextWrapper(width=80, initial_indent='', subsequent_indent='    ')

    def _get_or_download_model(self, base_model_name: str, quantization: str, repo_id: Optional[str]) -> Path:
        """Get model from local storage or download from HuggingFace."""
        # Construct full model filename
        model_filename = f"{base_model_name}.{quantization}.gguf"
        
        local_model = self.model_manager.get_model_path(model_filename)
        if local_model:
            print(f"Loading model from local storage: {local_model}")
            return local_model
        
        if not repo_id:
            raise ValueError("Model not found locally and no repo_id provided for download")
        
        print(f"Downloading model from HuggingFace: {repo_id}")
        
        # Verify the model exists in the repository
        available_models = self.model_manager.list_available_models(repo_id)
        if model_filename not in available_models:
            print("\nAvailable models:")
            for model in available_models:
                print(f"- {model}")
            raise ValueError(f"Model {model_filename} not found in repository. See available models above.")
        
        llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=model_filename,
            verbose=False
        )
        
        # Find the downloaded model in the cache
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        downloaded_model = None
        
        for path in cache_dir.rglob(model_filename):
            downloaded_model = path
            break
            
        if not downloaded_model:
            raise FileNotFoundError("Could not locate downloaded model in cache")
            
        # Save to our models directory
        return self.model_manager.save_model(downloaded_model, model_filename)

    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        """
        Generate and stream a response from the model.
        
        Args:
            prompt: User input prompt
            max_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            
        Returns:
            Complete response as a string
        """
        # Format conversation history into a single prompt
        full_prompt = self._format_conversation_history() + f"\nUser: {prompt}\nAssistant:"
        
        # Generate and stream response
        response_chunks = []
        print("\nAssistant: ", end="", flush=True)
        
        for chunk in self.llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=["User:"],
            stream=True
        ):
            chunk_text = chunk["choices"][0]["text"]
            print(chunk_text, end="", flush=True)
            response_chunks.append(chunk_text)
            
        print()  # New line after response
        return "".join(response_chunks).strip()

    def _format_conversation_history(self) -> str:
        """Format the conversation history into a prompt string."""
        formatted = []
        for entry in self.conversation_history:
            formatted.append(f"User: {entry['user']}")
            formatted.append(f"Assistant: {entry['assistant']}")
        return "\n".join(formatted)

    def chat(self):
        """Start the interactive chat interface."""
        print("\nWelcome to LlamaChat! Type 'quit' or 'exit' to end the conversation.")
        print("Type 'clear' to clear the conversation history.\n")
        
        while True:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            elif user_input.lower() == 'clear':
                self.conversation_history = []
                print("\nConversation history cleared.")
                continue
            elif not user_input:
                continue
            
            # Generate and display response
            try:
                response = self.generate_response(user_input)
                
                # Store the interaction in conversation history
                self.conversation_history.append({
                    'user': user_input,
                    'assistant': response
                })
                    
            except Exception as e:
                print(f"\nError: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Start a chat interface with a Llama model")
    parser.add_argument("--base-model-name", required=True, 
                      help="Base name of the model (e.g., 'llama-2-7b-chat')")
    parser.add_argument("--quantization", required=True,
                      help="Quantization format (e.g., 'Q4_0', 'Q5_K_M')")
    parser.add_argument("--repo-id", help="HuggingFace repo ID for downloading the model")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context window size")
    parser.add_argument("--n-threads", type=int, help="Number of threads to use")
    
    args = parser.parse_args()
    
    try:
        chat = LlamaChat(
            base_model_name=args.base_model_name,
            quantization=args.quantization,
            repo_id=args.repo_id,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads
        )
        chat.chat()
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
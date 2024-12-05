from pathlib import Path
import json
from dataclasses import dataclass
from typing import List, Optional
from llama_cpp import Llama
import os
from huggingface_hub import HfApi
from semantic_router import Route
from semantic_router.encoders import HuggingFaceEncoder
from semantic_router.layer import RouteLayer

class QueryRouter:
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.encoder = HuggingFaceEncoder(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Define routes for different query types
        programming_route = Route(
            name="programming",
            utterances=[
                "How do I write a function in Python?",
                "What's the syntax for a for loop?",
                "How to declare variables in JavaScript?",
                "Can you explain Object-Oriented Programming?",
                "How do I use try-except blocks?",
                "What are arrays in programming?",
                "How to implement data structures?",
                "Explain recursion in programming",
                "What is inheritance in OOP?",
                "How to handle exceptions in code?"
            ]
        )
        
        # Create the route layer
        self.router = RouteLayer(
            encoder=self.encoder,
            routes=[programming_route]
        )
    
    def is_programming_question(self, query: str) -> bool:
        """
        Determine if the input query is a programming-related question.
        """
        result = self.router(query)
        if self.debug:
            print(f"Router result: {result}")
        return result.name == "programming" if result else False

@dataclass
class ModelPromptFormat:
    system_prefix: str = ""
    system_suffix: str = ""
    user_prefix: str = "User: "
    user_suffix: str = ""
    assistant_prefix: str = "Assistant: "
    assistant_suffix: str = ""

@dataclass
class ModelConfig:
    model_name: str  # Simple display name (e.g., "qwen2.5-3b-instruct")
    model_file_name: str  # Actual GGUF file name (e.g., "qwen2.5-3b-instruct-q2_k.gguf")
    repo_id: str
    max_context_length: int
    stop_words: List[str]
    prompt_format: ModelPromptFormat
    system_prompt: Optional[str] = None

class ConfigManager:
    def __init__(self, config_path: str = "config.json", debug: bool = False):
        self.config_path = Path(config_path)
        self.debug = debug
        self._ensure_config()
        self.load_config()

    def _get_default_config(self):
        """Get the default configuration dictionary."""
        return {
            "models": [
                {
                    "model_name": "qwen2.5-3b-instruct",
                    "model_file_name": "qwen2.5-3b-instruct-q2_k.gguf",
                    "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
                    "max_context_length": 2048,
                    "stop_words": ["<|im_end|>"],
                    "system_prompt": "You are a helpful AI assistant.",
                    "prompt_format": {
                        "system_prefix": "<|im_start|>system\n",
                        "system_suffix": "<|im_end|>",
                        "user_prefix": "<|im_start|>user\n",
                        "user_suffix": "<|im_end|>",
                        "assistant_prefix": "<|im_start|>assistant\n",
                        "assistant_suffix": "<|im_end|>"
                    }
                }
            ]
        }

    def load_config(self):
        """Load the configuration from file."""
        with open(self.config_path) as f:
            self.config = json.load(f)

    def _ensure_config(self):
        """Ensure config file exists with default values."""
        if not self.config_path.exists():
            if self.debug:
                print("Config file not found. Creating default config...")
            self.create_default_config()
        else:
            # Validate existing config has all required fields
            try:
                with open(self.config_path) as f:
                    existing_config = json.load(f)
                default_config = self._get_default_config()
                
                # Check if any models are missing required fields
                updated = False
                for model in existing_config.get("models", []):
                    default_model = default_config["models"][0]
                    
                    # Handle migration from old config format
                    if "model_file_name" not in model and "model_name" in model:
                        if self.debug:
                            print(f"Migrating model config: {model.get('model_name', 'unknown')}")
                        model["model_file_name"] = model["model_name"]
                        model["model_name"] = model["model_file_name"].replace("-q2_k.gguf", "")
                        updated = True
                    
                    # Add any missing fields
                    for key, value in default_model.items():
                        if key not in model:
                            if self.debug:
                                print(f"Adding missing field '{key}' to model '{model.get('model_name', 'unknown')}'")
                            model[key] = value
                            updated = True
                
                if updated:
                    with open(self.config_path, "w") as f:
                        json.dump(existing_config, f, indent=2)
                        
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                if self.debug:
                    print(f"Error reading config file: {e}")
                    print("Creating new default config...")
                self.create_default_config()

    def get_model_config(self, model_name: Optional[str] = None) -> ModelConfig:
        if not self.config["models"]:
            raise ValueError("No models defined in config")
        
        if model_name:
            model_data = next((m for m in self.config["models"] if m["model_name"] == model_name), None)
            if not model_data:
                raise ValueError(f"Model {model_name} not found in config")
        else:
            model_data = self.config["models"][0]

        # Ensure all required fields exist
        default_model = self._get_default_config()["models"][0]
        for key, value in default_model.items():
            if key not in model_data:
                model_data[key] = value

        prompt_format = ModelPromptFormat(**model_data["prompt_format"])
        
        return ModelConfig(
            model_name=model_data["model_name"],
            model_file_name=model_data["model_file_name"],
            repo_id=model_data["repo_id"],
            max_context_length=model_data["max_context_length"],
            stop_words=model_data["stop_words"],
            prompt_format=prompt_format,
            system_prompt=model_data.get("system_prompt")
        )

class LlamaChat:
    def __init__(self, model_name: Optional[str] = None, n_threads: Optional[int] = None, debug: bool = False):
        self.debug = debug
        self.config_manager = ConfigManager(debug=debug)
        self.model_config = self.config_manager.get_model_config(model_name)
        self.query_router = QueryRouter(debug=debug)
        
        if self.debug:
            print(f"Loading model from {self.model_config.repo_id}")
            print(f"Using model file: {self.model_config.model_file_name}")
            
        self.llm = Llama.from_pretrained(
            repo_id=self.model_config.repo_id,
            filename=self.model_config.model_file_name,  # Use the full file name
            n_ctx=self.model_config.max_context_length,
            n_threads=n_threads or os.cpu_count(),
            verbose=debug
        )
        self.conversation_history = []

    def _format_prompt(self, user_input: str) -> str:
        pf = self.model_config.prompt_format
        
        formatted = []
        if self.model_config.system_prompt:
            formatted.append(f"{pf.system_prefix}{self.model_config.system_prompt}{pf.system_suffix}")
        
        for entry in self.conversation_history:
            formatted.append(f"{pf.user_prefix}{entry['user']}{pf.user_suffix}")
            formatted.append(f"{pf.assistant_prefix}{entry['assistant']}{pf.assistant_suffix}")
        
        formatted.append(f"{pf.user_prefix}{user_input}{pf.user_suffix}")
        formatted.append(f"{pf.assistant_prefix}")
        
        return "\n".join(formatted)

    def generate_response(self, prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
        full_prompt = self._format_prompt(prompt)
        
        response_chunks = []
        print("\nAssistant: ", end="", flush=True)
        
        for chunk in self.llm(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self.model_config.stop_words,
            stream=True
        ):
            chunk_text = chunk["choices"][0]["text"]
            print(chunk_text, end="", flush=True)
            response_chunks.append(chunk_text)
            
        print()
        return "".join(response_chunks).strip()

    def chat(self):
        print(f"\nWelcome to LlamaChat!" + (f" Using model: {self.model_config.model_name}" if self.debug else ""))
        print("Type 'quit' or 'exit' to end the conversation.")
        print("Type 'clear' to clear the conversation history.\n")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            elif user_input.lower() == 'clear':
                self.conversation_history = []
                print("\nConversation history cleared.")
                continue
            elif not user_input:
                continue
            
            try:
                # Check if it's a programming question
                if self.query_router.is_programming_question(user_input):
                    print("\nProgramming question!")
                
                response = self.generate_response(user_input)
                self.conversation_history.append({
                    'user': user_input,
                    'assistant': response
                })
            except Exception as e:
                print(f"\nError: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Start a chat interface with a Llama model")
    parser.add_argument("--model-name", help="Name of the model to use from config")
    parser.add_argument("--n-threads", type=int, help="Number of threads to use")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with verbose logging")
    
    args = parser.parse_args()
    
    try:
        chat = LlamaChat(
            model_name=args.model_name,
            n_threads=args.n_threads,
            debug=args.debug
        )
        chat.chat()
    except Exception as e:
        print(f"Error: {str(e)}")
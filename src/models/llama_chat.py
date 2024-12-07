import os
from typing import Optional, Dict
from llama_cpp import Llama
from config.config_manager import ConfigManager
from routing.query_router import QueryRouter

class LlamaChat:
    def __init__(self, n_threads: Optional[int] = None, debug: bool = False):
        self.debug = debug
        self.config_manager = ConfigManager(debug=debug)
        self.query_router = QueryRouter(debug=debug)
        
        # Load all models
        self.models: Dict[str, Llama] = {}
        self.load_models(n_threads)
        
        # Track current model for conversation
        self.current_model_group = None
        self.conversation_history = []

    def load_models(self, n_threads: Optional[int]):
        """Load all models from configured groups."""
        for group in self.config_manager.get_group_names():
            config = self.config_manager.get_model_config(group)
            if self.debug:
                print(f"Loading {group} model: {config.model_name}")
            
            self.models[group] = Llama.from_pretrained(
                repo_id=config.repo_id,
                filename=config.model_file_name,
                n_ctx=config.max_context_length,
                n_threads=n_threads or os.cpu_count(),
                verbose=self.debug
            )

    def _format_prompt(self, user_input: str, group: str) -> str:
        """Format prompt based on the selected model group."""
        config = self.config_manager.get_model_config(group)
        pf = config.prompt_format
        
        formatted = []
        if config.system_prompt:
            formatted.append(f"{pf.system_prefix}{config.system_prompt}{pf.system_suffix}")
        
        for entry in self.conversation_history:
            formatted.append(f"{pf.user_prefix}{entry['user']}{pf.user_suffix}")
            formatted.append(f"{pf.assistant_prefix}{entry['assistant']}{pf.assistant_suffix}")
        
        formatted.append(f"{pf.user_prefix}{user_input}{pf.user_suffix}")
        formatted.append(f"{pf.assistant_prefix}")
        
        return "\n".join(formatted)

    def generate_response(self, prompt: str, group: str = "default", max_tokens: int = 512, temperature: float = 0.7) -> str:
        model = self.models[group]
        config = self.config_manager.get_model_config(group)
        
        full_prompt = self._format_prompt(prompt, group)
        
        response_chunks = []
        print("\nAssistant: ", end="", flush=True)
        
        for chunk in model(
            full_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=config.stop_words,
            stream=True
        ):
            chunk_text = chunk["choices"][0]["text"]
            print(chunk_text, end="", flush=True)
            response_chunks.append(chunk_text)
            
        print()
        return "".join(response_chunks).strip()

    def chat(self):
        print("\nWelcome to LlamaChat!")
        print("Type 'quit' or 'exit' to end the conversation.")
        print("Type 'clear' to clear the conversation history.\n")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            elif user_input.lower() == 'clear':
                self.conversation_history = []
                self.current_model_group = None  # Reset model selection
                print("\nConversation history cleared.")
                continue
            elif not user_input:
                continue
            
            try:
                # If this is the first message or after a clear, determine the model
                if not self.current_model_group:
                    # Route to appropriate model based on query type
                    self.current_model_group = "programming" if self.query_router.is_programming_question(user_input) else "default"
                    if self.debug:
                        print(f"\nRouted to {self.current_model_group} model group")
                
                # Generate response using the selected model
                response = self.generate_response(user_input, self.current_model_group)
                self.conversation_history.append({
                    'user': user_input,
                    'assistant': response
                })
            except Exception as e:
                print(f"\nError: {str(e)}")
import os
from typing import Dict, Optional
from llama_cpp import Llama
from ..config.config_manager import ConfigManager
from ..routing.query_router import QueryRouter
from ..utils.prompt_formatter import PromptFormatter

class LlamaChat:
    def __init__(self, n_threads: Optional[int] = None, debug: bool = False):
        self.debug = debug
        self.config_manager = ConfigManager(debug=debug)
        self.query_router = QueryRouter(debug=debug)
        
        # Load models
        self.models: Dict[str, Llama] = {}
        self.model_configs = self.config_manager.get_model_configs()
        
        for model_name, config in self.model_configs.items():
            if self.debug:
                print(f"Loading model {model_name} from {config.repo_id}")
            
            self.models[model_name] = Llama.from_pretrained(
                repo_id=config.repo_id,
                filename=config.model_file_name,
                n_ctx=config.max_context_length,
                n_threads=n_threads or os.cpu_count(),
                verbose=debug
            )
        
        # Separate conversation histories for each model
        self.conversation_histories = {
            "qwen2.5-3b-instruct": [],
            "llama-3.2-3b-instruct": []
        }

    def generate_response(
        self,
        prompt: str,
        model_name: str,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        full_prompt = PromptFormatter.format_prompt(
            prompt,
            self.model_configs[model_name],
            self.conversation_histories[model_name]
        )
        
        model = self.models[model_name]
        config = self.model_configs[model_name]
        
        response_chunks = []
        print(f"\nAssistant ({model_name}): ", end="", flush=True)
        
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
        print("\nWelcome to LlamaChat!" + (" (Debug Mode)" if self.debug else ""))
        print("Type 'quit' or 'exit' to end the conversation.")
        print("Type 'clear' to clear the conversation history.\n")
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break
            elif user_input.lower() == 'clear':
                for history in self.conversation_histories.values():
                    history.clear()
                print("\nConversation history cleared.")
                continue
            elif not user_input:
                continue
            
            try:
                # Route to appropriate model based on query type
                is_programming = self.query_router.is_programming_question(user_input)
                model_name = "qwen2.5-3b-instruct" if is_programming else "llama-3.2-3b-instruct"
                
                if self.debug:
                    print(f"\nRouting to {model_name} {'(Programming)' if is_programming else '(General)'}")
                
                response = self.generate_response(user_input, model_name)
                self.conversation_histories[model_name].append({
                    'user': user_input,
                    'assistant': response
                })
            except Exception as e:
                print(f"\nError: {str(e)}")
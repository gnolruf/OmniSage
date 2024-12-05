import os
from typing import Optional

from llama_cpp import Llama

from config.config_manager import ConfigManager
from routing.query_router import QueryRouter


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
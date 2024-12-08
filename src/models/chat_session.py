import time
from typing import Dict, List, Optional, Union, Generator
from models.model_manager import ModelManager
from routing.query_router import QueryRouter

class ChatSession:
    def __init__(self, model_manager: ModelManager, debug: bool = False):
        """
        Initialize a chat session that will dynamically select its initial model.
        
        Args:
            model_manager: ModelManager instance with required models loaded
            debug: Enable debug output
        """
        self.model_manager = model_manager
        self.debug = debug
        self.query_router = QueryRouter(debug=debug)
        self.model_group: Optional[str] = None
        self.conversation_history: List[Dict[str, str]] = []

    def _select_model_group(self, query: str) -> str:
        """Use query router to select appropriate model group."""
        is_programming = self.query_router.is_programming_question(query)
        selected_group = "programming" if is_programming else "default"
        
        if not self.model_manager.is_model_loaded(selected_group):
            raise ValueError(f"Selected model group '{selected_group}' is not loaded. Please load it first.")
        
        if self.debug:
            print(f"\nSelected {selected_group} model based on query analysis")
            
        return selected_group

    def _format_prompt(self, user_input: str) -> str:
        """Format prompt based on the current model group."""
        if not self.model_group:
            raise ValueError("No model group selected. This shouldn't happen.")
            
        config = self.model_manager.get_config(self.model_group)
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

    def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 512, 
        temperature: float = 0.7,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """Generate a response using the current model with enhanced debugging."""
        if not self.model_group:
            self.model_group = self._select_model_group(prompt)
            
        model = self.model_manager.get_model(self.model_group)
        config = self.model_manager.get_config(self.model_group)
        
        full_prompt = self._format_prompt(prompt)
        
        if stream:
            def chunk_generator():
                previous_text = ""
                for chunk in model(
                    full_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=config.stop_words,
                    stream=True
                ):
                    chunk_text = chunk["choices"][0]["text"]
                    if self.debug:
                        print(f"Debug: Raw chunk received: {repr(chunk_text)}")
                    
                    # Only yield the new part of the text
                    new_text = chunk_text
                    if new_text:
                        if self.debug:
                            print(f"Debug: Yielding new text: {repr(new_text)}")
                        yield new_text
                        time.sleep(0.01)  # Small delay to help with streaming
            
            return chunk_generator()
        else:
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
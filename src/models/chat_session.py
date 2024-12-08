import time
from typing import Dict, List, Optional, Union, Generator
from models.model_manager import ModelManager
from routing.query_router import QueryRouter
from database_manager import DatabaseManager

class ChatSession:
    def __init__(
        self,
        model_manager: ModelManager,
        db_manager: DatabaseManager,
        chat_id: Optional[int] = None,
        debug: bool = False
    ):
        """
        Initialize a chat session with database support.
        
        Args:
            model_manager: ModelManager instance with required models loaded
            db_manager: DatabaseManager instance for persistence
            chat_id: Optional ID of existing chat to load
            debug: Enable debug output
        """
        self.model_manager = model_manager
        self.db_manager = db_manager
        self.debug = debug
        self.query_router = QueryRouter(debug=debug)
        self.model_group: Optional[str] = None
        self.conversation_history: List[Dict[str, str]] = []
        self.chat_id = chat_id
        
        # Load existing chat if chat_id provided
        if chat_id:
            self._load_chat_history(chat_id)
    
    def _select_model_group(self, query: str) -> str:
        """Use query router to select appropriate model group."""
        is_programming = self.query_router.is_programming_question(query)
        selected_group = "programming" if is_programming else "default"
        
        if not self.model_manager.is_model_loaded(selected_group):
            raise ValueError(f"Selected model group '{selected_group}' is not loaded. Please load it first.")
        
        if self.debug:
            print(f"\nSelected {selected_group} model based on query analysis")
            
        return selected_group

    def _load_chat_history(self, chat_id: int):
        """Load conversation history from database."""
        messages = self.db_manager.get_chat_messages(chat_id)
        self.conversation_history = [
            {msg["role"]: msg["content"]} for msg in messages
        ]
        
        # Set model_group from last assistant message if available
        assistant_messages = [
            msg for msg in messages 
            if msg["role"] == "assistant" and msg["model_group"]
        ]
        if assistant_messages:
            self.model_group = assistant_messages[-1]["model_group"]
    
    def _format_prompt(self, user_input: str) -> str:
        """Format prompt based on the current model group."""
        if not self.model_group:
            self.model_group = self._select_model_group(user_input)
            
        config = self.model_manager.get_config(self.model_group)
        pf = config.prompt_format
        
        formatted = []
        if config.system_prompt:
            formatted.append(f"{pf.system_prefix}{config.system_prompt}{pf.system_suffix}")
        
        # Format conversation history properly
        for entry in self.conversation_history:
            if "user" in entry:
                formatted.append(f"{pf.user_prefix}{entry['user']}{pf.user_suffix}")
            if "assistant" in entry:
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
                full_response = []
                for chunk in model(
                    full_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=config.stop_words,
                    stream=True
                ):
                    chunk_text = chunk["choices"][0]["text"]
                    if chunk_text:
                        full_response.append(chunk_text)
                        yield chunk_text
                        time.sleep(0.01)
                
                # Save to database if we have a chat_id
                if self.chat_id:
                    complete_response = "".join(full_response).strip()
                    self.db_manager.save_message(
                        self.chat_id,
                        "assistant",
                        complete_response,
                        self.model_group
                    )
                    
                    # Update conversation history
                    self.conversation_history.extend([
                        {"user": prompt},
                        {"assistant": complete_response}
                    ])
            
            return chunk_generator()
        else:
            response = model(
                full_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=config.stop_words,
                stream=False
            )
            complete_response = response["choices"][0]["text"].strip()
            
            # Save to database if we have a chat_id
            if self.chat_id:
                self.db_manager.save_message(
                    self.chat_id,
                    "assistant",
                    complete_response,
                    self.model_group
                )
            
            # Update conversation history
            self.conversation_history.extend([
                {"user": prompt},
                {"assistant": complete_response}
            ])
            
            return complete_response
import time
from typing import Dict, List, Optional, Union, Generator

from src.omnisage.models.manager import ModelManager
from src.omnisage.routers.query import QueryRouter
from src.omnisage.database.manager import DatabaseManager

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
        """Use query router to select appropriate model group, only on first message."""
        # If we already have a model group, keep using it
        if self.model_group:
            if self.debug:
                print(f"Using existing model group: {self.model_group}")
            return self.model_group

        is_programming = self.query_router.is_programming_question(query)
        selected_group = "programming" if is_programming else "default"
        
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
            
        config = self.model_manager.get_model_config(self.model_group)
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

    def _calculate_required_context(self, formatted_prompt: str, max_tokens: Optional[int] = None) -> tuple[int, int]:
        """
        Calculate required context size based on conversation and model parameters.
        Returns tuple of (context_size, output_tokens).
        """
        config = self.model_manager.get_model_config(self.model_group)
        
        # Estimate tokens in formatted prompt (simple estimation)
        estimated_prompt_tokens = len(formatted_prompt.split())
        
        # Get output tokens (use model's max_output_length if not specified)
        output_tokens = max_tokens if max_tokens is not None else config.max_output_length
        
        # Calculate total required tokens including space for response
        required_tokens = estimated_prompt_tokens + output_tokens
        
        # Calculate the minimum context size needed in multiples of max_output_length
        # that can fit our required tokens
        units_needed = (required_tokens + config.max_output_length - 1) // config.max_output_length
        context_size = units_needed * config.max_output_length
        
        if self.debug:
            print(f"Token calculation:")
            print(f"  - Prompt tokens: {estimated_prompt_tokens}")
            print(f"  - Output tokens: {output_tokens}")
            print(f"  - Total required: {required_tokens}")
            print(f"  - Units needed: {units_needed}")
            print(f"  - Context size: {context_size}")
        
        # Ensure we don't exceed the model's maximum context window
        context_size = min(context_size, config.context_window)
        
        # If our calculated context size can't fit the prompt plus output,
        # we need to reduce max output tokens
        if context_size < required_tokens:
            output_tokens = max(0, context_size - estimated_prompt_tokens)
            if self.debug:
                print(f"  - Adjusted output tokens: {output_tokens}")
        
        return context_size, output_tokens

    def generate_response(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """Generate a response using the current model with dynamic context management."""
        try:
            # Select model group if not already set
            if not self.model_group:
                self.model_group = self._select_model_group(prompt)
            
            # Get config for validation
            config = self.model_manager.get_model_config(self.model_group)
            
            # Format prompt for token calculation
            formatted_prompt = self._format_prompt(prompt)
            
            # Calculate required context size and output tokens
            required_ctx, output_tokens = self._calculate_required_context(formatted_prompt, max_tokens)
            
            if self.debug:
                print(f"Using context window size: {required_ctx}")
            
            # Get or load model - this will handle lazy loading internally
            model = self.model_manager.get_model(self.model_group)
            
            # Set context window size
            try:
                if self.debug:
                    print(f"Setting context window to: {required_ctx}")
                model.n_ctx = required_ctx
            except Exception as e:
                if self.debug:
                    print(f"Warning: Failed to update context window: {str(e)}")
            
            if self.debug:
                print(f"Using max_tokens: {output_tokens}")
                
            if stream:
                def chunk_generator():
                    full_response = []
                    try:
                        for chunk in model(
                            formatted_prompt,
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
                    except Exception as e:
                        if self.debug:
                            print(f"Error during streaming: {str(e)}")
                        raise
                
                return chunk_generator()
            else:
                response = model(
                    formatted_prompt,
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
                
        except Exception as e:
            if self.debug:
                print(f"Error generating response: {str(e)}")
            raise
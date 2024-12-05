from typing import List, Dict
from ..config.model_config import ModelConfig

class PromptFormatter:
    @staticmethod
    def format_prompt(
        user_input: str,
        config: ModelConfig,
        conversation_history: List[Dict[str, str]]
    ) -> str:
        pf = config.prompt_format
        formatted = []
        
        if config.system_prompt:
            formatted.append(f"{pf.system_prefix}{config.system_prompt}{pf.system_suffix}")
        
        for entry in conversation_history:
            formatted.append(f"{pf.user_prefix}{entry['user']}{pf.user_suffix}")
            formatted.append(f"{pf.assistant_prefix}{entry['assistant']}{pf.assistant_suffix}")
        
        formatted.append(f"{pf.user_prefix}{user_input}{pf.user_suffix}")
        formatted.append(f"{pf.assistant_prefix}")
        
        return "\n".join(formatted)
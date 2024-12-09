from typing import Dict, Optional
import os
from llama_cpp import Llama

from src.OmniSage.config.manager import ConfigManager

class ModelManager:
    def __init__(self, n_threads: Optional[int] = None, debug: bool = False):
        self.debug = debug
        self.config_manager = ConfigManager(debug=debug)
        self.n_threads = n_threads or os.cpu_count()
        self.models: Dict[str, Llama] = {}
    
    def load_model(self, group: str) -> None:
        """Load a specific model group."""
        if group not in self.config_manager.get_group_names():
            raise ValueError(f"Invalid model group: {group}")
            
        config = self.config_manager.get_model_config(group)
        if self.debug:
            print(f"Loading {group} model: {config.model_name}")
        
        self.models[group] = Llama.from_pretrained(
            repo_id=config.repo_id,
            filename=config.model_file_name,
            n_ctx=config.max_context_length,
            n_threads=self.n_threads,
            verbose=self.debug
        )
    
    def get_model(self, group: str) -> Llama:
        """Get a loaded model by group name."""
        if group not in self.models:
            raise ValueError(f"Model group '{group}' not loaded. Call load_model({group}) first.")
        return self.models[group]
    
    def get_config(self, group: str):
        """Get configuration for a model group."""
        return self.config_manager.get_model_config(group)
    
    def is_model_loaded(self, group: str) -> bool:
        """Check if a model group is loaded."""
        return group in self.models
    
    def available_groups(self) -> list[str]:
        """Get list of available model groups from config."""
        return self.config_manager.get_group_names()
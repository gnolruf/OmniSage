from typing import Dict, Optional
import os
from llama_cpp import Llama
from threading import RLock
from src.omnisage.config.manager import ConfigManager
from src.omnisage.models.config import ModelConfig

class ModelManager:
    def __init__(self, n_threads: Optional[int] = None, debug: bool = False):
        self.debug = debug
        self.config_manager = ConfigManager(debug=debug)
        self.n_threads = n_threads or os.cpu_count()
        self.models: Dict[str, Optional[Llama]] = {}
        self._locks: Dict[str, RLock] = {}
        
        # Initialize empty model slots for all configured groups
        for group in self.config_manager.get_group_names():
            self.models[group] = None
            self._locks[group] = RLock()
            
        # Preload default model
        self.preload_default_model()
    
    def get_model_config(self, group: str) -> ModelConfig:
        """Get configuration for a model group."""
        return self.config_manager.get_model_config(group)
    
    def preload_default_model(self):
        """Preload the default model at initialization."""
        try:
            if self.debug:
                print("Preloading default model...")
            
            config = self.get_model_config("default")
            min_output_length = self._get_min_output_length()
            
            self.models["default"] = Llama.from_pretrained(
                repo_id=config.repo_id,
                filename=config.model_file_name,
                n_ctx=min_output_length,  # Start with minimum output length
                n_threads=self.n_threads,
                verbose=self.debug,
                n_batch=512
            )
            
            if self.debug:
                print("Default model preloaded successfully")
                
        except Exception as e:
            error_msg = f"Failed to preload default model: {str(e)}"
            if self.debug:
                print(error_msg)
            # Don't raise the error - just log it and continue
            
    def _get_min_output_length(self) -> int:
        """Get the minimum max_output_length across all models."""
        min_length = float('inf')
        for group in self.config_manager.get_group_names():
            config = self.get_model_config(group)
            min_length = min(min_length, config.max_output_length)
        return min_length

    def ensure_model_loaded(self, group: str) -> None:
        """Ensure a specific model group is loaded, loading it if necessary."""
        if group not in self.config_manager.get_group_names():
            raise ValueError(f"Invalid model group: '{group}'. Available groups: {', '.join(self.config_manager.get_group_names())}")
        
        # Check if model needs to be loaded
        if self.models[group] is None:
            try:
                with self._locks[group]:
                    # Double-check after acquiring lock
                    if self.models[group] is None:
                        if self.debug:
                            print(f"Lazy loading {group} model...")
                        
                        config = self.get_model_config(group)
                        min_output_length = self._get_min_output_length()
                        
                        self.models[group] = Llama.from_pretrained(
                            repo_id=config.repo_id,
                            filename=config.model_file_name,
                            n_ctx=min_output_length,  # Start with minimum output length
                            n_threads=self.n_threads,
                            verbose=self.debug,
                            n_batch=512
                        )
            except Exception as e:
                error_msg = f"Failed to load {group} model: {str(e)}"
                if self.debug:
                    print(error_msg)
                raise RuntimeError(error_msg) from e
    
    def get_model(self, group: str) -> Llama:
        """Get a loaded model by group name, loading it if necessary."""
        self.ensure_model_loaded(group)
        return self.models[group]
    
    def is_model_loaded(self, group: str) -> bool:
        """Check if a model group is loaded."""
        return self.models.get(group) is not None
    
    def available_groups(self) -> list[str]:
        """Get list of available model groups from config."""
        return self.config_manager.get_group_names()
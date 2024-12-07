import json
from pathlib import Path
from typing import Optional, Dict

from config.model_config import ModelConfig, ModelPromptFormat

class ConfigManager:
    def __init__(self, config_dir: str = "configs", debug: bool = False):
        self.config_dir = Path(config_dir)
        self.models_dir = self.config_dir / "models"
        self.router_dir = self.config_dir / "routers"
        self.debug = debug
        self._ensure_config_structure()
        self.load_config()

    def _ensure_config_structure(self):
        """Ensure config directory structure exists."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.router_dir.mkdir(parents=True, exist_ok=True)

        # Ensure models config exists
        models_config = self.models_dir / "models.json"
        if not models_config.exists():
            if self.debug:
                print("Creating default models config...")
            self._create_default_models_config(models_config)

    def _create_default_models_config(self, config_path: Path):
        """Create a default models configuration file."""
        default_config = {
            "model_groups": {
                "default": {
                    "model_name": "llama-3.2-3b-instruct",
                    "model_file_name": "llama-3.2-3b-instruct-q2_k.gguf",
                    "repo_id": "unsloth/Llama-3.2-3B-Instruct-GGUF",
                    "max_context_length": 2048,
                    "stop_words": ["<|eot_id|>"],
                    "system_prompt": "You are a helpful AI assistant. You provide accurate, helpful responses in a clear and natural way.",
                    "prompt_format": {
                        "system_prefix": "<|start_header_id|>system<|end_header_id|>\n",
                        "system_suffix": "<|eot_id|>",
                        "user_prefix": "<|start_header_id|>user<|end_header_id|>\n",
                        "user_suffix": "<|eot_id|>",
                        "assistant_prefix": "<|start_header_id|>assistant<|end_header_id|>\n",
                        "assistant_suffix": "<|eot_id|>"
                    }
                },
                "programming": {
                    "model_name": "qwen2.5-3b-instruct",
                    "model_file_name": "qwen2.5-3b-instruct-q2_k.gguf",
                    "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
                    "max_context_length": 2048,
                    "stop_words": ["<|im_end|>"],
                    "system_prompt": "You are a helpful programming assistant focused on providing accurate and detailed coding help.",
                    "prompt_format": {
                        "system_prefix": "<|im_start|>system\n",
                        "system_suffix": "<|im_end|>",
                        "user_prefix": "<|im_start|>user\n",
                        "user_suffix": "<|im_end|>",
                        "assistant_prefix": "<|im_start|>assistant\n",
                        "assistant_suffix": "<|im_end|>"
                    }
                }
            }
        }

        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)

    def load_config(self):
        """Load models configuration."""
        config_path = self.models_dir / "models.json"
        if not config_path.exists():
            raise ValueError("Models configuration file not found")
            
        with open(config_path) as f:
            self.config = json.load(f)
            
        self.model_groups = self.config["model_groups"]

    def get_model_config(self, group: str) -> ModelConfig:
        """Get model configuration for a specific group."""
        if group not in self.model_groups:
            raise ValueError(f"Model group '{group}' not found in configurations")
            
        model_data = self.model_groups[group]
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

    def get_group_names(self) -> list[str]:
        """Get list of available model group names."""
        return list(self.model_groups.keys())

    def save_model_config(self, config: dict):
        """Save the entire models configuration to file."""
        config_path = self.models_dir / "models.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        self.load_config()
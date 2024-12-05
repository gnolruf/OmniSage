import json
from pathlib import Path
from typing import Optional

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
        # Create directory structure if it doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.router_dir.mkdir(parents=True, exist_ok=True)

        # Ensure default model config exists
        default_model_config = self.models_dir / "default.json"
        if not default_model_config.exists():
            if self.debug:
                print("Creating default model config...")
            self._create_default_model_config(default_model_config)

    def _create_default_model_config(self, config_path: Path):
        """Create a default model configuration file."""
        default_config = {
            "model_name": "qwen2.5-3b-instruct",
            "model_file_name": "qwen2.5-3b-instruct-q2_k.gguf",
            "repo_id": "Qwen/Qwen2.5-3B-Instruct-GGUF",
            "max_context_length": 2048,
            "stop_words": ["<|im_end|>"],
            "system_prompt": "You are a helpful AI assistant.",
            "prompt_format": {
                "system_prefix": "<|im_start|>system\n",
                "system_suffix": "<|im_end|>",
                "user_prefix": "<|im_start|>user\n",
                "user_suffix": "<|im_end|>",
                "assistant_prefix": "<|im_start|>assistant\n",
                "assistant_suffix": "<|im_end|>"
            }
        }

        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)

    def load_config(self):
        """Load all model configurations."""
        self.models = {}
        for config_file in self.models_dir.glob("*.json"):
            with open(config_file) as f:
                model_config = json.load(f)
                self.models[model_config["model_name"]] = model_config

    def get_model_config(self, model_name: Optional[str] = None) -> ModelConfig:
        """Get a specific model configuration or the default one."""
        if not self.models:
            raise ValueError("No model configurations found")

        if model_name:
            if model_name not in self.models:
                raise ValueError(f"Model {model_name} not found in configurations")
            model_data = self.models[model_name]
        else:
            # Use the first available model as default
            model_data = next(iter(self.models.values()))

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

    def save_model_config(self, model_name: str, config: dict):
        """Save a model configuration to file."""
        config_path = self.models_dir / f"{model_name}.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        self.load_config()  # Reload configurations
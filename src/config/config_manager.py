import json
from pathlib import Path
from typing import Optional

from config.model_config import ModelConfig, ModelPromptFormat


class ConfigManager:
    def __init__(self, config_path: str = "config.json", debug: bool = False):
        self.config_path = Path(config_path)
        self.debug = debug
        self._ensure_config()
        self.load_config()

    def _get_default_config(self):
        """Get the default configuration dictionary."""
        return {
            "models": [
                {
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
            ]
        }

    def load_config(self):
        """Load the configuration from file."""
        with open(self.config_path) as f:
            self.config = json.load(f)

    def _ensure_config(self):
        """Ensure config file exists with default values."""
        if not self.config_path.exists():
            if self.debug:
                print("Config file not found. Creating default config...")
            self.create_default_config()
        else:
            # Validate existing config has all required fields
            try:
                with open(self.config_path) as f:
                    existing_config = json.load(f)
                default_config = self._get_default_config()
                
                # Check if any models are missing required fields
                updated = False
                for model in existing_config.get("models", []):
                    default_model = default_config["models"][0]
                    
                    # Handle migration from old config format
                    if "model_file_name" not in model and "model_name" in model:
                        if self.debug:
                            print(f"Migrating model config: {model.get('model_name', 'unknown')}")
                        model["model_file_name"] = model["model_name"]
                        model["model_name"] = model["model_file_name"].replace("-q2_k.gguf", "")
                        updated = True
                    
                    # Add any missing fields
                    for key, value in default_model.items():
                        if key not in model:
                            if self.debug:
                                print(f"Adding missing field '{key}' to model '{model.get('model_name', 'unknown')}'")
                            model[key] = value
                            updated = True
                
                if updated:
                    with open(self.config_path, "w") as f:
                        json.dump(existing_config, f, indent=2)
                        
            except (json.JSONDecodeError, KeyError, IndexError) as e:
                if self.debug:
                    print(f"Error reading config file: {e}")
                    print("Creating new default config...")
                self.create_default_config()

    def create_default_config(self):
        """Create a new config file with default values."""
        with open(self.config_path, "w") as f:
            json.dump(self._get_default_config(), f, indent=2)

    def get_model_config(self, model_name: Optional[str] = None) -> ModelConfig:
        if not self.config["models"]:
            raise ValueError("No models defined in config")
        
        if model_name:
            model_data = next((m for m in self.config["models"] if m["model_name"] == model_name), None)
            if not model_data:
                raise ValueError(f"Model {model_name} not found in config")
        else:
            model_data = self.config["models"][0]

        # Ensure all required fields exist
        default_model = self._get_default_config()["models"][0]
        for key, value in default_model.items():
            if key not in model_data:
                model_data[key] = value

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
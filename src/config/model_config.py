from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ModelPromptFormat:
    system_prefix: str = ""
    system_suffix: str = ""
    user_prefix: str = "User: "
    user_suffix: str = ""
    assistant_prefix: str = "Assistant: "
    assistant_suffix: str = ""

@dataclass
class ModelConfig:
    model_name: str
    model_file_name: str
    repo_id: str
    max_context_length: int
    stop_words: List[str]
    prompt_format: ModelPromptFormat
    system_prompt: Optional[str] = None
<div align="center">

# OmniSage
### A Framework for Serving Local LLM Ensembles

</div>

---

OmniSage is a powerful framework designed for serving multiple local language models (LLMs) with intelligent routing capabilities. It supports running multiple models simultaneously, with each model specialized for specific types of queries, backed by a PostgreSQL database for conversation persistence.

## Prerequisites

* Python 3.8+
* PostgreSQL 12+
* Required Python packages (see `requirements.txt`)
* Local LLM models in GGUF format

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/omnisage.git
   cd omnisage
   ```

2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Set up the database:
   ```sh
   python setup_db.py --username your_username --password your_password
   ```

## Configuration

### Database Configuration
The database configuration is stored in `configs/database.json`:

```json
{
    "host": "localhost",
    "port": 5432,
    "database": "llamachat",
    "user": "your_username",
    "password": "your_password"
}
```

### Model Configuration
Model configurations are stored in `configs/models/models.json`. Example configuration:

```json
{
    "model_groups": {
        "default": {
            "model_name": "llama-3.2-3b-instruct",
            "model_file_name": "llama-3.2-3b-instruct-q2_k.gguf",
            "repo_id": "unsloth/Llama-3.2-3B-Instruct-GGUF",
            "max_context_length": 2048,
            "stop_words": ["<|eot_id|>"],
            "system_prompt": "You are a helpful AI assistant...",
            "prompt_format": {
                "system_prefix": "<|start_header_id|>system<|end_header_id|>\n",
                "system_suffix": "<|eot_id|>",
                "user_prefix": "<|start_header_id|>user<|end_header_id|>\n",
                "user_suffix": "<|eot_id|>",
                "assistant_prefix": "<|start_header_id|>assistant<|end_header_id|>\n",
                "assistant_suffix": "<|eot_id|>"
            }
        }
    }
}
```

### Router Configuration
Query routing configurations are stored in `configs/routers/default.json`:

```json
{
    "encoder_type": "huggingface",
    "encoder_name": "all-MiniLM-L6-v2",
    "routes": [
        {
            "name": "programming",
            "utterances": [
                "How do I write a function in Python?",
                "What's the syntax for a for loop?"
            ],
            "description": "Programming and software development related questions",
            "score_threshold": 0.3
        }
    ]
}
```

## Usage

### Starting the API Server

```sh
python -m src.OmniSage.api.app
```

The API server will start on `http://localhost:8000` by default.

### Using the CLI Interface

```sh
python -m src.OmniSage.cli.main
```

Available commands in CLI:

| Command | Description |
|---------|-------------|
| `/new <title>` | Create new chat |
| `/load` | List and load saved chats |
| `/delete` | Delete a chat |
| `/clear` | Clear current conversation |
| `/quit` | Exit the program |

### API Endpoints

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/chat/stream` | POST | Stream chat responses |
| `/chat/chats` | POST | Create a new chat session |
| `/chat/chats` | GET | List all chat sessions |
| `/chat/chats/{chat_id}/messages` | GET | Get messages for a specific chat |
| `/chat/chats/{chat_id}` | DELETE | Delete a chat session |

## Architecture

1. **Model Manager**: Handles loading and managing multiple LLM models
2. **Query Router**: Routes incoming queries to appropriate models
3. **Chat Session**: Manages conversation state and model interactions
4. **Database Manager**: Handles persistence of conversations and messages
5. **API Layer**: Provides REST API endpoints for external integration
6. **CLI Interface**: Provides interactive command-line interface

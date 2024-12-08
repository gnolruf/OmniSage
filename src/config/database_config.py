from pathlib import Path
import json
from typing import Optional, Dict
from functools import lru_cache

class DatabaseConfig:
    """Singleton configuration class for database settings."""
    _instance = None

    def __new__(cls, config_dir: str = "configs"):
        if cls._instance is None:
            cls._instance = super(DatabaseConfig, cls).__new__(cls)
            cls._instance.config_dir = Path(config_dir)
            cls._instance.db_config_file = cls._instance.config_dir / "database.json"
            cls._instance._ensure_config()
            cls._instance.load_config()
        return cls._instance

    def _ensure_config(self):
        """Ensure database configuration file exists."""
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        if not self.db_config_file.exists():
            default_config = {
                "host": "localhost",
                "port": 5432,
                "database": "llamachat",
                "user": "postgres",
                "password": ""  # Should be set by user
            }
            
            with open(self.db_config_file, "w") as f:
                json.dump(default_config, f, indent=2)
    
    @lru_cache()
    def load_config(self) -> Dict:
        """Load database configuration with caching."""
        with open(self.db_config_file) as f:
            return json.load(f)
    
    def get_connection_string(self) -> str:
        """Get PostgreSQL connection string from config."""
        config = self.load_config()
        return (
            f"postgresql://{config['user']}:{config['password']}"
            f"@{config['host']}:{config['port']}/{config['database']}"
        )

    def update_config(self, **kwargs):
        """Update database configuration and clear cache."""
        config = self.load_config()
        config.update(kwargs)
        
        with open(self.db_config_file, "w") as f:
            json.dump(config, f, indent=2)
        
        # Clear the cache to reload config
        self.load_config.cache_clear()
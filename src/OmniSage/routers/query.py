from pathlib import Path
import json
from semantic_router import Route, RouteLayer
from semantic_router.encoders import HuggingFaceEncoder

class QueryRouter:
    def __init__(self, config_dir: str = "configs", debug: bool = False):
        self.debug = debug
        self.config_dir = Path(config_dir)
        self.router_dir = self.config_dir / "routers"
        
        # Initialize encoder
        self.encoder = HuggingFaceEncoder(
            model_name="all-MiniLM-L6-v2"
        )
        
        # Ensure config structure exists
        self._ensure_config_structure()
        
        # Load routes from config
        self.router = self._load_route_layer()
    
    def _ensure_config_structure(self):
        """Ensure router config directory and default config exist."""
        self.router_dir.mkdir(parents=True, exist_ok=True)
        
        default_config_path = self.router_dir / "default.json"
        if not default_config_path.exists():
            if self.debug:
                print("Creating default router config...")
            self._create_default_config(default_config_path)
    
    def _create_default_config(self, config_path: Path):
        """Create a default router configuration file."""
        default_config = {
            "encoder_type": "huggingface",
            "encoder_name": "all-MiniLM-L6-v2",
            "routes": [
                {
                    "name": "programming",
                    "utterances": [
                        "How do I write a function in Python?",
                        "What's the syntax for a for loop?",
                        "How to declare variables in JavaScript?",
                        "Can you explain Object-Oriented Programming?",
                        "How do I use try-except blocks?",
                        "What are arrays in programming?",
                        "How to implement data structures?",
                        "Explain recursion in programming",
                        "What is inheritance in OOP?",
                        "How to handle exceptions in code?"
                    ],
                    "description": "Programming and software development related questions",
                    "score_threshold": 0.3
                }
            ]
        }
        
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
    
    def _load_route_layer(self) -> RouteLayer:
        """Load routes from config and initialize RouteLayer."""
        config_files = list(self.router_dir.glob("*.json"))
        if not config_files:
            raise ValueError("No router configurations found")
        
        # Load and merge all route configurations
        all_routes = []
        for config_file in config_files:
            with open(config_file) as f:
                config = json.load(f)
                routes = [Route(**route) for route in config["routes"]]
                all_routes.extend(routes)
        
        if self.debug:
            print(f"Loaded {len(all_routes)} routes from {len(config_files)} config files")
        
        return RouteLayer(
            encoder=self.encoder,
            routes=all_routes
        )
    
    def save_routes(self, filename: str, routes: list[dict]):
        """Save routes to a configuration file."""
        config_path = self.router_dir / filename
        config = {
            "encoder_type": "huggingface",
            "encoder_name": "all-MiniLM-L6-v2",
            "routes": routes
        }
        
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        # Reload routes
        self.router = self._load_route_layer()
    
    def is_programming_question(self, query: str) -> bool:
        """Determine if the input query is a programming-related question."""
        result = self.router(query)
        if self.debug:
            print(f"Router result: {result}")
        return result.name == "programming" if result else False
    
    def get_route(self, query: str):
        """Get the matching route for a query."""
        return self.router(query)
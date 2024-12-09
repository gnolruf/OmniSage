from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.OmniSage.models.manager import ModelManager
from src.OmniSage.database.manager import DatabaseManager
from src.OmniSage.database.config import DatabaseConfig

# Global instances
model_manager = None
db_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    print("Initializing services...")
    
    # Initialize model manager and load models
    app.state.model_manager = ModelManager(debug=False)
    print("Loading models...")
    for group in app.state.model_manager.available_groups():
        print(f"Loading {group} model...")
        app.state.model_manager.load_model(group)
    
    # Initialize database connection
    db_config = DatabaseConfig()
    app.state.db_manager = DatabaseManager(db_config.get_connection_string())
    
    yield
    
    # Cleanup
    print("Shutting down...")
    if app.state.db_manager:
        app.state.db_manager.close()

# Create FastAPI application
app = FastAPI(
    title="Omnisage API",
    description="AI Chat API with multiple model support",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Import and include routers after app creation
from src.OmniSage.api.routes import chat
app.include_router(chat.router)

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()

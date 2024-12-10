from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.omnisage.models.manager import ModelManager
from src.omnisage.database.manager import DatabaseManager
from src.omnisage.database.config import DatabaseConfig

# Global instances
model_manager = None
db_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    print("Initializing services...")
    
    # Initialize model manager without loading models
    app.state.model_manager = ModelManager(debug=False)
    print("Model manager initialized (models will be loaded on demand)")
    
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
from src.omnisage.api.routes import chat
app.include_router(chat.router)

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start_server()

"""FastAPI application setup and configuration."""

import logging
from fastapi import FastAPI

from src.api.routers import health, chat
from src.api.middleware.cors import setup_cors
from src.api.middleware.logging import LoggingMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create FastAPI app
app = FastAPI(
    title="RAG Assistant API",
    description="Engineering Knowledge RAG Assistant API",
    version="0.1.0",
)

# Setup middleware
setup_cors(app)
app.add_middleware(LoggingMiddleware)

# Include routers
app.include_router(health.router)
app.include_router(chat.router)


def run_app(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Run the FastAPI application using uvicorn.
    
    Args:
        host: Host to bind to
        port: Port to bind to
        reload: Enable auto-reload for development
    """
    import uvicorn
    uvicorn.run(
        "src.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


if __name__ == "__main__":
    run_app(reload=True)


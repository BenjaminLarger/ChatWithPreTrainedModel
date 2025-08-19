"""
Main application entry point for the Multi-Model Chat Interface
"""
import uvicorn
from src.utils.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run(
        "src.api.routes:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
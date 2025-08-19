"""
Configuration settings for the Multi-Model Chat Interface using Pydantic
"""
import os
from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    
    # Hugging Face Configuration
    huggingface_api_token: Optional[str] = Field(None, env="HUGGINGFACE_API_TOKEN")
    
    # Model Configuration
    model_cache_dir: str = Field("./model_cache", env="MODEL_CACHE_DIR")
    
    # Application Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    debug: bool = Field(False, env="DEBUG")
    
    # API Configuration
    host: str = Field("localhost", env="HOST")
    port: int = Field(8000, env="PORT")
    
    # Model Settings
    default_model: str = Field("openai", env="DEFAULT_MODEL")
    max_tokens: int = Field(150, env="MAX_TOKENS")
    temperature: float = Field(0.7, env="TEMPERATURE")
    
    # Chat Settings
    max_context_length: int = Field(4000, env="MAX_CONTEXT_LENGTH")
    conversation_timeout: int = Field(3600, env="CONVERSATION_TIMEOUT")  # 1 hour in seconds
    
    # Model-specific Settings
    distilbert_model_name: str = Field("distilbert-base-uncased", env="DISTILBERT_MODEL_NAME")
    huggingface_model_name: str = Field("microsoft/DialoGPT-medium", env="HUGGINGFACE_MODEL_NAME")
    
    # Performance Settings
    batch_size: int = Field(1, env="BATCH_SIZE")
    use_gpu: bool = Field(False, env="USE_GPU")
    num_workers: int = Field(1, env="NUM_WORKERS")
    
    # Security Settings
    api_rate_limit: int = Field(100, env="API_RATE_LIMIT")  # requests per minute
    max_message_length: int = Field(2000, env="MAX_MESSAGE_LENGTH")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


def create_model_cache_dir() -> str:
    """Create model cache directory if it doesn't exist"""
    settings = get_settings()
    cache_dir = settings.model_cache_dir
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir, exist_ok=True)
    
    return cache_dir


def get_model_config() -> dict:
    """Get model configuration dictionary"""
    settings = get_settings()
    
    return {
        "openai": {
            "api_key": settings.openai_api_key,
            "max_tokens": settings.max_tokens,
            "temperature": settings.temperature,
            "model": "gpt-3.5-turbo"
        },
        "huggingface": {
            "api_token": settings.huggingface_api_token,
            "model_name": settings.huggingface_model_name,
            "cache_dir": settings.model_cache_dir,
            "use_gpu": settings.use_gpu
        },
        "distilbert": {
            "model_name": settings.distilbert_model_name,
            "cache_dir": settings.model_cache_dir,
            "use_gpu": settings.use_gpu,
            "max_length": 512
        }
    }
# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Multi-Model Chat Interface project that integrates three types of AI models:
- **OpenAI API** (GPT-3.5-turbo) for advanced text generation
- **Hugging Face transformers** for open-source model access
- **DistilBERT** for text classification and lightweight generation

The project is built with FastAPI for the backend API, includes a web-based chat interface, and supports fine-tuning capabilities for custom datasets.

## Development Setup

### Virtual Environment
Always work within the virtual environment located at `.venv/`:
```bash
source .venv/bin/activate  # Linux/Mac
```

### Environment Variables
Create a `.env` file with required API keys:
```
OPENAI_API_KEY=your_openai_api_key_here
HUGGINGFACE_API_TOKEN=your_hf_token_here
MODEL_CACHE_DIR=./model_cache
```

## Common Development Commands

### Running the Application
```bash
python main.py
```
This starts the FastAPI server using uvicorn with settings from `src.utils.config`.

### Testing
```bash
# Test OpenAI client functionality
python tests/test_openai_client.py

# Run all tests with pytest
pytest tests/
```

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Development Server
```bash
# Run with auto-reload for development
uvicorn src.api.routes:app --reload --host localhost --port 8000
```

### Jupyter Notebooks
```bash
# Start Jupyter for fine-tuning experiments
jupyter notebook notebooks/
```

## Architecture Overview

### Core Components
- **`src/models/`**: Model clients for OpenAI, Hugging Face, and DistilBERT
- **`src/chat/`**: Chat handling and message processing logic
- **`src/api/`**: FastAPI routes and endpoints
- **`src/utils/`**: Configuration management using Pydantic Settings

### Configuration System
The project uses Pydantic Settings with environment variable support in `src/utils/config.py`. Settings are cached using `@lru_cache()` for performance. Key configuration categories:
- **Model configurations**: tokens, temperature, cache directories
- **API settings**: host, port, rate limits  
- **Chat settings**: context length, timeouts
- **Performance settings**: GPU usage, batch size, workers
- **Security settings**: rate limits, message length validation

### Model Integration Pattern
Each model type follows a consistent pattern:
1. Client initialization with settings from config
2. Error handling with exponential backoff retry logic
3. Connection validation methods
4. Standardized response format with usage metrics

### Frontend Structure
- **`templates/`**: Jinja2 HTML templates for the web interface
- **`static/`**: CSS and JavaScript for the chat interface
- WebSocket support for real-time chat

## Key Implementation Details

### OpenAI Client (`src/models/openai_client.py`)
- Implements retry logic with exponential backoff for rate limits
- Supports conversation history context (last 10 messages)
- Includes comprehensive error handling for different API exceptions
- Tracks token usage and model performance

### Configuration Management
The `get_settings()` function provides cached access to all configuration values. The `get_model_config()` function returns model-specific configurations. Model cache directories are automatically created via `create_model_cache_dir()` when needed.

### Error Handling Strategy
- Rate limiting handled with exponential backoff
- Input validation for prompt length and content
- Comprehensive logging at different levels
- Graceful degradation when APIs are unavailable

## Development Workflow

1. **Environment Setup**: Activate virtual environment and create `.env` file with required API keys
2. **Dependencies**: Install requirements with `pip install -r requirements.txt`
3. **Testing**: Run individual model tests (`python tests/test_openai_client.py`) before integration
4. **Development**: Use `python main.py` for production or `uvicorn` with `--reload` for development
5. **Model Validation**: Test API connections individually before full system integration
6. **Jupyter Development**: Use `notebooks/` for experimentation and fine-tuning

## Project Structure Notes

- **Modular Architecture**: Clear separation between models, chat logic, API routes, and utilities
- **Model Abstraction**: Clients provide uniform interfaces for different AI services (OpenAI, Hugging Face, DistilBERT)
- **Configuration Management**: Centralized Pydantic settings with environment variable support and caching
- **Cache Management**: Model files cached in `model_cache/` directory, automatically managed
- **Testing Strategy**: Individual client tests with direct Python execution (not pytest framework)
- **Frontend Integration**: Static files and Jinja2 templates for web interface
- **Development Tools**: Jupyter notebooks in `notebooks/` for experimentation and fine-tuning

## Important Implementation Details

### Model Cache Directory
Models are automatically downloaded and cached in `model_cache/`. This directory can become large (several GB) and should be excluded from version control.

### Test Execution Pattern
Tests are designed to run as standalone Python scripts with direct execution (`python tests/test_*.py`) rather than through pytest, allowing for interactive testing and validation of API connections.

### Environment Configuration
All model-specific settings (API keys, model names, cache paths) are managed through the centralized config system in `src/utils/config.py` with environment variable overrides.
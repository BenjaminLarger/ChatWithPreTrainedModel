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

## Architecture Overview

### Core Components
- **`src/models/`**: Model clients for OpenAI, Hugging Face, and DistilBERT
- **`src/chat/`**: Chat handling and message processing logic
- **`src/api/`**: FastAPI routes and endpoints
- **`src/utils/`**: Configuration management using Pydantic Settings

### Configuration System
The project uses Pydantic Settings with environment variable support in `src/utils/config.py`. Key settings include:
- Model configurations (tokens, temperature, cache directories)
- API settings (host, port, rate limits)
- Chat settings (context length, timeouts)
- Performance settings (GPU usage, batch size)

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
The `get_settings()` function provides cached access to all configuration values. Model cache directories are automatically created when needed.

### Error Handling Strategy
- Rate limiting handled with exponential backoff
- Input validation for prompt length and content
- Comprehensive logging at different levels
- Graceful degradation when APIs are unavailable

## Development Workflow

1. **Environment Setup**: Activate virtual environment and set up `.env` file
2. **Testing**: Run individual model tests before full integration
3. **Development**: Use the FastAPI development server with reload enabled
4. **Model Testing**: Use the test scripts in `tests/` to validate API connections

## Project Structure Notes

- The project follows a modular architecture separating concerns
- Models are abstracted behind client interfaces for easy swapping
- Configuration is centralized and environment-aware
- Chat functionality is separated from model implementations
- Frontend and backend are loosely coupled through API endpoints
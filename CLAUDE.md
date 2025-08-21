# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Multi-Model Chat Interface project that integrates four types of AI models:
- **OpenAI API** (GPT-3.5-turbo) for advanced text generation
- **Hugging Face transformers** for local open-source model access
- **Hugging Face Inference API** for serverless model access without local downloads
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
# Test individual model clients
python tests/test_openai_client.py
python tests/test_huggingface_client.py
python tests/test_inference_api_client.py

# Test specific models directly with quick validation
python test_specific_models.py

# Comprehensive model comparison (downloads multiple models, generates reports)
python tests/test_model_comparison.py

# Run all tests with pytest (if using pytest framework)
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
- **`src/models/`**: Model clients for OpenAI, Hugging Face (local + Inference API), and DistilBERT
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

### Inference API Client (`src/models/inference_api_client.py`)
- Serverless access to Hugging Face models without local downloads
- Supports multiple NLP tasks: text generation, classification, Q&A, summarization, embeddings
- Built-in retry logic and error handling for API failures
- Rate limit management and model loading detection
- No local storage requirements - ideal for production deployments

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
3. **Quick Model Testing**: Run `python test_specific_models.py` to validate core models quickly
4. **Individual Client Testing**: Run individual model tests (`python tests/test_openai_client.py`) for specific clients
5. **Comprehensive Testing**: Use `python tests/test_model_comparison.py` for full model performance analysis
6. **Development**: Use `python main.py` for production or `uvicorn` with `--reload` for development
7. **Model Validation**: Test API connections individually before full system integration
8. **Jupyter Development**: Use `notebooks/` for experimentation and fine-tuning

## Project Structure Notes

- **Modular Architecture**: Clear separation between models, chat logic, API routes, and utilities
- **Model Abstraction**: Clients provide uniform interfaces for different AI services (OpenAI, local Hugging Face, Inference API, DistilBERT)
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

### Model Testing Infrastructure
- **`test_specific_models.py`**: Quick validation of specific Hugging Face models with immediate feedback
- **`tests/test_model_comparison.py`**: Comprehensive performance comparison of multiple models with detailed metrics and reporting
- **`tests/test_inference_api_client.py`**: Complete testing suite for Inference API functionality across multiple NLP tasks
- **Test Reports**: Automatically generated JSON data and markdown reports in `tests/reports/` directory

### Model Client Comparison
- **Local Hugging Face** (`huggingface_client.py`): Full model downloads, offline usage, complete control, high disk usage
- **Inference API** (`inference_api_client.py`): Serverless, no downloads, internet required, potential rate limits, production-ready
- **OpenAI API** (`openai_client.py`): Premium service, reliable, cost per usage, conversation context support
- **DistilBERT** (`distilbert_model.py`): Lightweight, fast classification, local processing

### Environment Configuration
All model-specific settings (API keys, model names, cache paths) are managed through the centralized config system in `src/utils/config.py` with environment variable overrides.
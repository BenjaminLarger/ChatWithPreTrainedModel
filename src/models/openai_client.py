from src.utils.config import get_settings
from openai import OpenAI, OpenAIError, RateLimitError, APITimeoutError, APIConnectionError
import logging
import time
from typing import Optional, Dict, Any
from functools import wraps


def retry_with_exponential_backoff(
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple = (RateLimitError, APITimeoutError, APIConnectionError),
):
    """Decorator for retrying OpenAI API calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            num_retries = 0
            delay = initial_delay

            while True:
                try:
                    return func(*args, **kwargs)
                except errors as e:
                    num_retries += 1
                    if num_retries > max_retries:
                        raise e
                    
                    # Add jitter to prevent thundering herd
                    import random
                    if jitter:
                        delay *= exponential_base * (0.5 + random.random() * 0.5)
                    else:
                        delay *= exponential_base
                    
                    args[0].logger.warning(f"API call failed, retrying in {delay:.2f} seconds. Error: {str(e)}")
                    time.sleep(delay)
                except Exception as e:
                    # Don't retry for non-recoverable errors
                    raise e
        return wrapper
    return decorator


class OpenAIClient:
    def __init__(self):
        self.settings = get_settings()
        self.api_key = self.settings.openai_api_key
        self.model = "gpt-3.5-turbo"  # Use specific model instead of default_model
        self.max_tokens = self.settings.max_tokens
        self.temperature = self.settings.temperature
        
        try:
            self.client = OpenAI(api_key=self.api_key)
        except Exception as e:
            raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")
        
        # Set up logging
        log_level = getattr(logging, self.settings.log_level.upper(), logging.INFO)
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"OpenAI client initialized with model: {self.model}")

    @retry_with_exponential_backoff()
    def generate_text(self, prompt: str, conversation_history: Optional[list] = None) -> Dict[str, Any]:
        """
        Generate text using OpenAI API with error handling and retry logic
        
        Args:
            prompt: User input prompt
            conversation_history: Optional conversation context
            
        Returns:
            Dict containing response text, usage info, and metadata
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if len(prompt) > self.settings.max_message_length:
            raise ValueError(f"Prompt too long. Maximum length: {self.settings.max_message_length}")
        
        # Build messages array
        messages = []
        if conversation_history:
            messages.extend(conversation_history[-10:])  # Keep last 10 messages for context
        
        messages.append({
            "role": "user",
            "content": prompt
        })
        
        try:
            self.logger.info(f"Generating text for prompt: {prompt[:100]}...")
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                timeout=30  # 30 second timeout
            )
            
            # Extract response data
            result = {
                "text": response.choices[0].message.content,
                "model": response.model,
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                },
                "finish_reason": response.choices[0].finish_reason,
                "created": response.created
            }
            
            self.logger.info(f"Successfully generated response. Tokens used: {result['usage']['total_tokens']}")
            self.logger.debug(f"Full response: {result}")
            
            return result
            
        except RateLimitError as e:
            self.logger.error(f"Rate limit exceeded: {str(e)}")
            raise
        except APITimeoutError as e:
            self.logger.error(f"API timeout: {str(e)}")
            raise
        except APIConnectionError as e:
            self.logger.error(f"API connection error: {str(e)}")
            raise
        except OpenAIError as e:
            self.logger.error(f"OpenAI API error: {str(e)}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error in text generation: {str(e)}")
            raise

    def get_available_models(self) -> list:
        """Get list of available models"""
        try:
            models = self.client.models.list()
            return [model.id for model in models.data if 'gpt' in model.id]
        except Exception as e:
            self.logger.error(f"Failed to fetch available models: {str(e)}")
            return []

    def validate_connection(self) -> bool:
        """Test if the OpenAI connection is working"""
        try:
            self.client.models.list()
            self.logger.info("OpenAI connection validated successfully")
            return True
        except Exception as e:
            self.logger.error(f"OpenAI connection validation failed: {str(e)}")
            return False

"""
Hugging Face Inference API Client
Provides access to hosted models via HTTP API endpoints without local downloads
"""
import requests
import json
import logging
import time
from typing import Optional, Dict, Any, List, Union
from functools import wraps
from src.utils.config import get_settings
from huggingface_hub import InferenceClient
import os

def retry_with_exponential_backoff(
    initial_delay: float = 1.0,
    exponential_base: float = 2.0,
    jitter: bool = True,
    max_retries: int = 3,
    errors: tuple = (requests.exceptions.RequestException, requests.exceptions.Timeout),
):
    """Decorator for retrying API calls with exponential backoff"""
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


class InferenceAPIClient:
    """
    Hugging Face Inference API client for serverless model inference
    Supports text generation, classification, and other NLP tasks via HTTP API
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.api_token = self.settings.huggingface_api_token
        self.base_url = "https://api-inference.huggingface.co/models"
        
        # Request configuration
        self.timeout = 30
        self.max_retries = 3
        
        # Default models for different tasks
        self.default_models = {
            "text-generation": "openai/gpt-oss-120b",
            "text-classification": "cardiffnlp/twitter-roberta-base-sentiment-latest",
            # "question-answering": "distilbert-base-cased-distilled-squad",
            # "summarization": "facebook/bart-large-cnn",
            # "translation": "t5-base",
            # "text2text-generation": "t5-base",
            # "feature-extraction": "sentence-transformers/all-MiniLM-L6-v2"
        }
        
        # Set up logging
        log_level = getattr(logging, self.settings.log_level.upper(), logging.INFO)
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Prepare headers
        self.headers = {
            "Content-Type": "application/json",
        }
        if self.api_token:
            self.headers["Authorization"] = f"Bearer {self.api_token}"
            self.logger.info("Inference API client initialized with authentication")
        else:
            self.logger.warning("No API token provided - using free tier with rate limits")
        
        self.logger.info(f"Inference API client initialized")
    
    @retry_with_exponential_backoff()
    def generate_text(
        self, 
        prompt: str, 
        model_name: Optional[str] = None,
        max_new_tokens: int = None,
        temperature: float = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using Inference API
        
        Args:
            prompt: Input text prompt
            model_name: Model to use (default: gpt2)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional generation parameters
            
        Returns:
            Dict containing generated text and metadata
        """
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if len(prompt) > self.settings.max_message_length:
            raise ValueError(f"Prompt too long. Maximum length: {self.settings.max_message_length}")
        
        model_name = model_name or self.default_models["text-generation"]

        try:
          self.logger.info(f"Generating text with model {model_name} for prompt: {prompt[:100]}...")
          client = InferenceClient(
              provider="cerebras",
              api_key=os.environ["HUGGINGFACE_API_TOKEN"],
          )
          completion = client.chat.completions.create(
              model=model_name,
              messages=[
                  {
                      "role": "user",
                      "content": prompt
                  }
              ]
          )
          print(f"completion: {completion.choices[0].message}")
          generated_text = completion.choices[0].message.content
          input_tokens = len(prompt.split())
          output_tokens = len(generated_text.split())
          
          result = {
              "text": generated_text,
              "model": model_name,
              "usage": {
                  "prompt_tokens": input_tokens,
                  "completion_tokens": output_tokens,
                  "total_tokens": input_tokens + output_tokens
              },
              "parameters": kwargs,
              "api_type": "inference_api"
          }
        except Exception as e:
            self.logger.error(f"Text generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate text: {str(e)}")

        self.logger.info(f"Text generation successful. Estimated tokens: {result['usage']['total_tokens']}")
        return result
            
    @retry_with_exponential_backoff()
    def classify_text(
        self, 
        text: str, 
        model_name: Optional[str] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Classify text using Inference API
        
        Args:
            text: Text to classify
            model_name: Model to use for classification
            top_k: Number of top predictions to return
            
        Returns:
            Dict containing classification results
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        model_name = model_name or self.default_models["text-classification"]
        

        try:
            self.logger.info(f"Classifying text with model {model_name}: {text[:100]}...")
            
            client = InferenceClient(
                provider="hf-inference",
                api_key=os.environ["HUGGINGFACE_API_TOKEN"],
            )

            response = client.text_classification(
                "I like you. I love you",
                model="tabularisai/multilingual-sentiment-analysis",
            )
            
            result = {
                "text": text,
                "predictions": response,
                "model": model_name,
                "api_type": "inference_api"
            }
            
            self.logger.info("Text classification successful")
            return result
            
        except Exception as e:
            self.logger.error(f"Text classification failed: {str(e)}")
            raise RuntimeError(f"Failed to classify text: {str(e)}")
    
    
    def validate_connection(self) -> bool:
        """Test if the Inference API connection is working"""
        try:
            # Test with a simple text generation
            test_result = self.generate_text("Hello", max_new_tokens=5)
            
            if test_result and "text" in test_result:
                self.logger.info("Inference API connection validated successfully")
                return True
            else:
                self.logger.error("Inference API validation failed: No text in result")
                return False
                
        except Exception as e:
            self.logger.error(f"Inference API validation failed: {str(e)}")
            return False
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about available models and API status"""
        model_name = model_name or self.default_models["text-generation"]
        
        return {
            "api_type": "inference_api",
            "base_url": self.base_url,
            "authenticated": bool(self.api_token),
            "default_models": self.default_models,
            "selected_model": model_name,
            "timeout": self.timeout,
            "max_retries": self.max_retries
        }
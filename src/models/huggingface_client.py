from src.utils.config import get_settings, create_model_cache_dir
from transformers import (
    pipeline
)
import torch
import logging
from typing import Optional, Dict, Any, List
from functools import lru_cache
import warnings

# Suppress transformers warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")


class HuggingFaceClient:
    """
    Hugging Face transformers client with support for text generation and classification
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.model_name = self.settings.huggingface_model_name
        self.cache_dir = create_model_cache_dir()
        self.use_gpu = self.settings.use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # Initialize logging
        log_level = getattr(logging, self.settings.log_level.upper(), logging.INFO)
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Pipeline storage
        self._generation_pipeline = None
        self._classification_pipeline = None
        self._tokenizer = None
        
        self.logger.info(f"HuggingFace client initialized")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Cache directory: {self.cache_dir}")
        
        # Initialize default generation pipeline
        self._initialize_generation_pipeline()
    def _initialize_generation_pipeline(self) -> None:
        """Initialize the text generation pipeline"""
        try:
            self.logger.info(f"Loading text generation pipeline for {self.model_name}")
            
            # Create pipeline with configuration
            self._generation_pipeline = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
                device=0 if self.use_gpu else -1,  # 0 for first GPU, -1 for CPU
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                trust_remote_code=True,
                return_full_text=False,  # Return only generated text
                do_sample=True,
                temperature=self.settings.temperature,
                max_new_tokens=self.settings.max_tokens,
                pad_token_id=50256  # GPT-2 pad token
            )
            
            self.logger.info(" Text generation pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize generation pipeline: {str(e)}")
            self.logger.info("Attempting to load with fallback configuration...")
            
            try:
                # Fallback: simpler configuration
                self._generation_pipeline = pipeline(
                    "text-generation",
                    model=self.model_name,
                    device=-1,  # Force CPU
                )
                self.logger.info(" Generation pipeline initialized with fallback config")
                self.use_gpu = False
                self.device = "cpu"
                
            except Exception as fallback_error:
                self.logger.error(f"Fallback initialization also failed: {str(fallback_error)}")
                self._generation_pipeline = None
                raise RuntimeError(f"Could not initialize Hugging Face pipeline: {str(fallback_error)}")
    
    def _initialize_classification_pipeline(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest") -> None:
        """Initialize text classification pipeline"""
        try:
            self.logger.info(f"Loading classification pipeline for {model_name}")
            
            self._classification_pipeline = pipeline(
                "text-classification",
                model=model_name,
                device=0 if self.use_gpu else -1,
                return_all_scores=True,
            )
            
            self.logger.info(" Classification pipeline initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize classification pipeline: {str(e)}")
            self._classification_pipeline = None
    
    def generate_text(self, prompt: str, max_length: Optional[int] = None, **kwargs) -> Dict[str, Any]:
        """
        Generate text using the Hugging Face pipeline
        
        Args:
            prompt: Input text prompt
            max_length: Maximum length of generated text
            **kwargs: Additional generation parameters
            
        Returns:
            Dict containing generated text and metadata
        """
        if not self._generation_pipeline:
            raise RuntimeError("Generation pipeline not initialized")
        
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty")
        
        if len(prompt) > self.settings.max_message_length:
            raise ValueError(f"Prompt too long. Maximum length: {self.settings.max_message_length}")
        
        try:
            self.logger.info(f"Generating text for prompt: {prompt[:100]}...")
            
            # Set generation parameters
            generation_kwargs = {
                "max_new_tokens": max_length or self.settings.max_tokens,
                "temperature": kwargs.get("temperature", self.settings.temperature),
                "do_sample": kwargs.get("do_sample", True),
                "top_p": kwargs.get("top_p", 0.9),
                "top_k": kwargs.get("top_k", 50),
                "repetition_penalty": kwargs.get("repetition_penalty", 1.1),
                "pad_token_id": self._generation_pipeline.tokenizer.eos_token_id
            }
            
            # Generate text
            outputs = self._generation_pipeline(
                prompt,
                **generation_kwargs
            )
            
            # Extract generated text
            if isinstance(outputs, list) and len(outputs) > 0:
                generated_text = outputs[0]["generated_text"]
            else:
                generated_text = str(outputs)
            
            # Calculate token usage (approximate)
            input_tokens = len(self._generation_pipeline.tokenizer.encode(prompt))
            output_tokens = len(self._generation_pipeline.tokenizer.encode(generated_text))
            
            result = {
                "text": generated_text,
                "model": self.model_name,
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                },
                "device": self.device,
                "parameters": generation_kwargs
            }
            
            self.logger.info(f"Text generation successful. Tokens: {result['usage']['total_tokens']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Text generation failed: {str(e)}")
            raise RuntimeError(f"Failed to generate text: {str(e)}")
    
    def classify_text(self, text: str, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Classify text using Hugging Face classification models
        
        Args:
            text: Text to classify
            model_name: Optional specific model for classification
            
        Returns:
            Dict containing classification results
        """
        # Initialize classification pipeline if needed
        if not self._classification_pipeline or model_name:
            self._initialize_classification_pipeline(model_name or "cardiffnlp/twitter-roberta-base-sentiment-latest")
        
        if not self._classification_pipeline:
            raise RuntimeError("Classification pipeline not available")
        
        try:
            self.logger.info(f"Classifying text: {text[:100]}...")
            
            results = self._classification_pipeline(text)
            
            # Format results
            classification_result = {
                "text": text,
                "predictions": results,
                "model": model_name or "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "device": self.device
            }
            
            self.logger.info("Text classification successful")
            return classification_result
            
        except Exception as e:
            self.logger.error(f"Text classification failed: {str(e)}")
            raise RuntimeError(f"Failed to classify text: {str(e)}")
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """Get information about available model types"""
        return {
            "text_generation": [
                "facebook/blenderbot_small-90M",
                "gpt2",

            ],
            "text_classification": [
                "cardiffnlp/twitter-roberta-base-sentiment-latest",
                "distilbert-base-uncased-finetuned-sst-2-english",
            ]
        }
    
    def validate_connection(self) -> bool:
        """Test if the Hugging Face pipeline is working"""
        try:
            if not self._generation_pipeline:
                return False
                
            # Test with a simple prompt
            test_result = self.generate_text("Hello", max_length=10)
            
            if test_result and "text" in test_result:
                self.logger.info("HuggingFace pipeline validation successful")
                return True
            else:
                self.logger.error("HuggingFace pipeline validation failed: No text in result")
                return False
                
        except Exception as e:
            self.logger.error(f"HuggingFace pipeline validation failed: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        if not self._generation_pipeline:
            return {"error": "No pipeline loaded"}
        
        try:
            model_config = self._generation_pipeline.model.config
            tokenizer_info = self._generation_pipeline.tokenizer
            
            return {
                "model_name": self.model_name,
                "model_type": getattr(model_config, 'model_type', 'unknown'),
                "vocab_size": getattr(model_config, 'vocab_size', 'unknown'),
                "max_position_embeddings": getattr(model_config, 'max_position_embeddings', 'unknown'),
                "device": self.device,
                "tokenizer_type": type(tokenizer_info).__name__,
                "special_tokens": {
                    "bos_token": tokenizer_info.bos_token,
                    "eos_token": tokenizer_info.eos_token,
                    "pad_token": tokenizer_info.pad_token,
                    "unk_token": tokenizer_info.unk_token
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get model info: {str(e)}")
            return {"error": str(e)}
    
    def unload_model(self) -> None:
        """Unload the model to free memory"""
        try:
            if self._generation_pipeline:
                del self._generation_pipeline
                self._generation_pipeline = None
            
            if self._classification_pipeline:
                del self._classification_pipeline  
                self._classification_pipeline = None
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self.logger.info("Models unloaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error unloading models: {str(e)}")


@lru_cache()
def get_huggingface_client() -> HuggingFaceClient:
    """Get cached HuggingFace client instance"""
    return HuggingFaceClient()
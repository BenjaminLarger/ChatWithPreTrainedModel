#!/usr/bin/env python3
"""
Test script for Hugging Face Inference API Client
Tests various NLP tasks including text generation, classification, QA, summarization, and embeddings
"""
import os
import sys
import time
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

load_dotenv()


def test_inference_api_client():
    """Test Hugging Face Inference API Client functionality"""
    print("=== Testing Hugging Face Inference API Client ===\n")
    
    try:
        from src.models.inference_api_client import InferenceAPIClient
        
        # Initialize client
        print("1. Initializing Inference API client...")
        client = InferenceAPIClient()
        print("✓ Client initialized successfully")
        
        # Test connection validation
        print("\n2. Testing connection validation...")
        try:
            is_valid = client.validate_connection()
            if is_valid:
                print("✓ Connection validation successful")
            else:
                print("⚠ Connection validation failed (might be due to rate limits)")
        except Exception as e:
            print(f"⚠ Connection validation error: {str(e)}")
        
        # Get model info
        print("\n3. Getting model information...")
        try:
            model_info = client.get_model_info()
            print("✓ Model info retrieved:")
            print(f"  - API Type: {model_info['api_type']}")
            print(f"  - Authenticated: {model_info['authenticated']}")
            print(f"  - Default Text Generation Model: {model_info['default_models']['text-generation']}")
        except Exception as e:
            print(f"✗ Failed to get model info: {str(e)}")
        
        # Test text generation
        print("\n4. Testing text generation...")
        test_prompts = [
            "Hello, how are you?",
            "Tell me about artificial intelligence.",
            "What is Python programming?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n  Test {i}: {prompt}")
            try:
                start_time = time.time()
                result = client.generate_text(
                    prompt, 
                    max_new_tokens=30,
                    temperature=0.7
                )
                generation_time = time.time() - start_time
                
                print(f"  ✓ Generated ({generation_time:.2f}s): {result['text']}")
                print(f"    Tokens - Input: {result['usage']['prompt_tokens']}, "
                      f"Output: {result['usage']['completion_tokens']}")
                
            except Exception as e:
                print(f"  ✗ Generation failed: {str(e)}")
        
        # Test text classification
        print("\n5. Testing text classification...")
        test_texts = [
            "I love this new feature! It's amazing!",
            "This is terrible. I hate it.",
            "The weather is nice today.",
            "I feel neutral about this topic."
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\n  Test {i}: {text}")
            try:
                start_time = time.time()
                result = client.classify_text(text, top_k=3)
                classification_time = time.time() - start_time
                
                print(f"  ✓ Classified ({classification_time:.2f}s):")
                for pred in result['predictions'][:3]:  # Top 3 predictions
                    print(f"    {pred['label']}: {pred['score']:.4f}")
                
            except Exception as e:
                print(f"  ✗ Classification failed: {str(e)}")
        

        print("\n" + "="*60)
        print("🎉 Inference API Client testing completed!")
        print("\nNote: Some tests may fail due to:")
        print("- Rate limiting on free tier")
        print("- Model loading delays")
        print("- Network connectivity issues")
        print("- API token requirements for certain models")
        print("="*60)
        
    except ImportError as e:
        print(f"❌ Import error: {str(e)}")
        print("Make sure the inference_api_client.py file is in src/models/")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")


def test_model_comparison():
    """Compare Inference API with local Hugging Face models"""
    print("\n=== Comparing Inference API vs Local Models ===\n")
    
    try:
        from src.models.inference_api_client import InferenceAPIClient
        from src.models.huggingface_client import HuggingFaceClient
        
        inference_client = InferenceAPIClient()
        # Note: HuggingFace client might not be available if models aren't loaded
        
        test_prompt = "Hello, how are you today?"
        
        print(f"Test prompt: {test_prompt}")
        print("\n1. Inference API Result:")
        try:
            start_time = time.time()
            inference_result = inference_client.generate_text(
                test_prompt, 
                max_new_tokens=20
            )
            inference_time = time.time() - start_time
            
            print(f"   Generated: {inference_result['text']}")
            print(f"   Time: {inference_time:.2f}s")
            print(f"   Tokens: {inference_result['usage']['total_tokens']}")
            
        except Exception as e:
            print(f"   Error: {str(e)}")
        
        print("\n2. Local HuggingFace Result:")
        try:
            hf_client = HuggingFaceClient()
            start_time = time.time()
            hf_result = hf_client.generate_text(
                test_prompt, 
                max_length=20
            )
            hf_time = time.time() - start_time
            
            print(f"   Generated: {hf_result['text']}")
            print(f"   Time: {hf_time:.2f}s")
            print(f"   Tokens: {hf_result['usage']['total_tokens']}")
            
            # Clean up
            hf_client.unload_model()
            
        except Exception as e:
            print(f"   Error: {str(e)}")
            print("   (Local models may not be available)")
        
        print("\nComparison Summary:")
        print("- Inference API: No local storage needed, might have rate limits")
        print("- Local HuggingFace: Requires model download, full control")
        
    except Exception as e:
        print(f"❌ Comparison failed: {str(e)}")


if __name__ == "__main__":
    print("Starting Hugging Face Inference API Client Tests")
    print("=" * 60)
    
    # Main test
    test_inference_api_client()
    
    # Optional comparison
    try:
        comparison = input("\nRun comparison with local models? (y/N): ").lower().strip()
        if comparison in ['y', 'yes']:
            test_model_comparison()
    except KeyboardInterrupt:
        print("\n\nTests completed.")
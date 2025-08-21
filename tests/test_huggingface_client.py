#!/usr/bin/env python3
"""
Test script for Hugging Face client with transformers pipeline
"""
import os
import sys
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

load_dotenv()

def test_huggingface_client():
    """Test Hugging Face client functionality"""
    print("=== Hugging Face Client Test ===\n")
    
    try:
        from src.models.huggingface_client import HuggingFaceClient
        
        # Initialize client
        print("1. Initializing Hugging Face client...")
        client = HuggingFaceClient()
        print("✓ Client initialized successfully\n")
        
        # Test connection validation
        print("2. Validating pipeline...")
        is_valid = client.validate_connection()
        if is_valid:
            print("✓ Pipeline validation successful\n")
        else:
            print("✗ Pipeline validation failed\n")
            return False
        
        # Get model information
        print("3. Getting model information...")
        model_info = client.get_model_info()
        if "error" not in model_info:
            print("✓ Model info retrieved:")
            print(f"  Model: {model_info.get('model_name', 'Unknown')}")
            print(f"  Type: {model_info.get('model_type', 'Unknown')}")
            print(f"  Device: {model_info.get('device', 'Unknown')}")
            print(f"  Vocab size: {model_info.get('vocab_size', 'Unknown')}\n")
        else:
            print(f"⚠ Could not get model info: {model_info.get('error')}\n")
        
        # Test text generation
        print("4. Testing text generation...")
        test_prompt = "Hello! How are you?"
        
        try:
            response = client.generate_text(test_prompt, max_length=50)
            print("✓ Text generation successful!")
            print(f"Prompt: {test_prompt}")
            print(f"Generated: {response['text'][:200]}...")
            print(f"Tokens used: {response['usage']['total_tokens']}")
            print(f"Model: {response['model']}")
            print(f"Device: {response['device']}\n")
        except Exception as e:
            print(f"✗ Text generation failed: {str(e)}\n")
            return False
        
        # Test text classification
        print("5. Testing text classification...")
        test_text = "I love this new feature! It's amazing!"
        
        try:
            classification_result = client.classify_text(test_text)
            print("✓ Text classification successful!")
            print(f"Text: {test_text}")
            print("Predictions:")
            # Handle both possible return types: list or dict with 'predictions'
            predictions = classification_result
            if isinstance(classification_result, dict) and 'predictions' in classification_result:
                predictions = classification_result['predictions']
            predictions = predictions[:3]
            for pred in predictions:
                print(f"  - {pred['label']}: {pred['score']:.4f}")
            if isinstance(classification_result, dict) and 'model' in classification_result:
                print(f"Model: {classification_result['model']}\n")
            else:
                print()
        except Exception as e:
            print(f"⚠ Text classification failed: {str(e)}")
            print("This is optional functionality\n")
        
        # Test error handling
        print("6. Testing error handling...")
        try:
            client.generate_text("")  # Empty prompt
            print("✗ Should have failed with empty prompt")
            return False
        except ValueError as e:
            print("✓ Empty prompt validation works")
            print(f"  Error: {str(e)}\n")
        
        # Test long prompt handling
        try:
            long_prompt = "This is a test. " * 200  # Very long prompt
            client.generate_text(long_prompt)
            print("⚠ Long prompt was accepted (may be valid depending on model)")
        except ValueError as e:
            print("✓ Long prompt validation works")
            print(f"  Error: {str(e)}\n")
        
        # Get available models
        print("7. Getting available models...")
        available_models = client.get_available_models()
        print("✓ Available model categories:")
        for category, models in available_models.items():
            print(f"  {category}: {len(models)} models")
            for model in models[:2]:  # Show first 2 models
                print(f"    - {model}")
        print()
        
        print("=== Most tests passed! ===")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {str(e)}")
        print("Make sure all dependencies are installed:")
        print("pip install transformers torch")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
        return False


def test_memory_management():
    """Test model loading and unloading"""
    print("\n=== Testing Memory Management ===")
    
    try:
        from src.models.huggingface_client import HuggingFaceClient
        
        print("1. Loading model...")
        client = HuggingFaceClient()
        
        print("2. Testing model functionality...")
        result = client.generate_text("Test", max_length=10)
        if result:
            print("✓ Model working correctly")
        
        print("3. Unloading model...")
        client.unload_model()
        print("✓ Model unloaded successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Memory management test failed: {str(e)}")
        return False


if __name__ == "__main__":
    print("Starting Hugging Face client tests...\n")
    
    success = test_huggingface_client()
    
    if success:
        memory_success = test_memory_management()
        if memory_success:
            print("\n🎉 All Hugging Face client tests passed!")
        else:
            print("\n⚠ Core functionality works, but memory management had issues")
    else:
        print("\n❌ Some core tests failed. Check the error messages above.")
        print("\nNote: First run may take longer due to model downloads.")
        sys.exit(1)
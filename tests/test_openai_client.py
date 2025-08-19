#!/usr/bin/env python3
"""
Test script for OpenAI client with error handling and retry logic
"""
import os
import sys
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

load_dotenv()

def test_openai_client():
    """Test OpenAI client functionality"""
    print("=== OpenAI Client Test ===\n")
    
    try:
        from src.models.openai_client import OpenAIClient
        
        # Initialize client
        print("1. Initializing OpenAI client...")
        client = OpenAIClient()
        print("✓ Client initialized successfully\n")
        
        # Test connection validation
        print("2. Validating connection...")
        is_valid = client.validate_connection()
        if is_valid:
            print("✓ Connection validated successfully\n")
        else:
            print("✗ Connection validation failed\n")
            return False
        
        # Test available models
        print("3. Fetching available models...")
        models = client.get_available_models()
        if models:
            print(f"✓ Found {len(models)} models:")
            for model in models[:5]:  # Show first 5 models
                print(f"  - {model}")
            print()
        else:
            print("⚠ No models found or API error\n")
        
        # Test text generation with simple prompt
        print("4. Testing text generation...")
        test_prompt = "Hello! Please respond with a short greeting."
        
        try:
            response = client.generate_text(test_prompt)
            print("✓ Text generation successful!")
            print(f"Response: {response['text']}")
            print(f"Tokens used: {response['usage']['total_tokens']}")
            print(f"Model: {response['model']}")
            print(f"Finish reason: {response['finish_reason']}\n")
        except Exception as e:
            print(f"✗ Text generation failed: {str(e)}\n")
            return False
        
        # Test with conversation history
        print("5. Testing with conversation history...")
        conversation = [
            {"role": "user", "content": "What's your name?"},
            {"role": "assistant", "content": "I'm ChatGPT, an AI assistant."},
        ]
        
        try:
            response = client.generate_text("Tell me a fun fact.", conversation)
            print("✓ Conversation context test successful!")
            print(f"Response: {response['text'][:100]}...")
            print(f"Tokens used: {response['usage']['total_tokens']}\n")
        except Exception as e:
            print(f"✗ Conversation context test failed: {str(e)}\n")
            return False
        
        # Test error handling with invalid input
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
            long_prompt = "x" * 3000  # Very long prompt
            client.generate_text(long_prompt)
            print("✗ Should have failed with long prompt")
            return False
        except ValueError as e:
            print("✓ Long prompt validation works")
            print(f"  Error: {str(e)}\n")
        
        print("=== All tests passed! ===")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {str(e)}")
        print("Make sure all dependencies are installed:")
        print("pip install openai python-dotenv pydantic")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {str(e)}")
        return False


def test_without_api_key():
    """Test behavior when API key is missing"""
    print("\n=== Testing without API key ===")
    
    # Temporarily remove API key
    original_key = os.environ.get('OPENAI_API_KEY')
    if original_key:
        del os.environ['OPENAI_API_KEY']
    
    try:
        from models.openai_client import OpenAIClient
        client = OpenAIClient()
        print("✗ Should have failed without API key")
        return False
    except Exception as e:
        print("✓ Properly handles missing API key")
        print(f"  Error: {str(e)}")
        
        # Restore API key
        if original_key:
            os.environ['OPENAI_API_KEY'] = original_key
        return True


if __name__ == "__main__":
    # Check if .env file exists
    if not os.path.exists('.env'):
        print("⚠ Warning: .env file not found")
        print("Create .env file with OPENAI_API_KEY=your_key_here")
        print("Using .env.example as reference\n")
    
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY') or os.getenv('OPENAI_API_KEY') == 'your_openai_api_key_here':
        print("⚠ Warning: OPENAI_API_KEY not set or using placeholder value")
        print("Set your actual OpenAI API key in the .env file\n")
        print("Skipping API tests...")
        sys.exit(1)
    
    success = test_openai_client()
    
    if success:
        print("\n🎉 OpenAI client is working correctly!")
    else:
        print("\n❌ Some tests failed. Check the error messages above.")
        sys.exit(1)
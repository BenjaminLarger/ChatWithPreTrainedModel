#!/usr/bin/env python3
"""
Quick test script for specific Hugging Face models requested
"""
import os
import sys
import time
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

load_dotenv()

def test_text_generation_model(model_name, display_name):
    """Test a specific text generation model"""
    print(f"\n=== Testing {display_name} ===")
    print(f"Model: {model_name}")
    
    try:
        from transformers import pipeline
        import torch
        
        print("Loading model... (this may take a while for first run)")
        start_time = time.time()
        
        # Create pipeline
        generator = pipeline(
            "text-generation",
            model=model_name,
            device=-1,  # CPU for compatibility
            trust_remote_code=True
        )
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f} seconds")
        
        # Test with sample prompts
        test_prompts = [
            "Hello, how are you?",
            "Tell me about AI",
            "What is Python programming?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt}")
            
            try:
                start_time = time.time()
                response = generator(
                    prompt,
                    max_new_tokens=50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=generator.tokenizer.eos_token_id
                )
                generation_time = time.time() - start_time
                
                # Extract generated text
                if isinstance(response, list) and len(response) > 0:
                    generated = response[0]['generated_text']
                    # Remove the input prompt to show only generated part
                    generated = generated[len(prompt):].strip()
                else:
                    generated = str(response)
                
                print(f"Generated ({generation_time:.2f}s): {generated}")
                
            except Exception as e:
                print(f"✗ Generation failed: {str(e)}")
        
        # Clean up
        del generator
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"✓ {display_name} test completed")
        return True
        
    except Exception as e:
        print(f"✗ Failed to load {display_name}: {str(e)}")
        return False


def test_classification_model(model_name, display_name):
    """Test a specific text classification model"""
    print(f"\n=== Testing {display_name} ===")
    print(f"Model: {model_name}")
    
    try:
        from transformers import pipeline
        
        print("Loading model... (this may take a while for first run)")
        start_time = time.time()
        
        # Create pipeline
        classifier = pipeline(
            "text-classification",
            model=model_name,
            device=-1,  # CPU for compatibility
            return_all_scores=True
        )
        
        load_time = time.time() - start_time
        print(f"✓ Model loaded in {load_time:.2f} seconds")
        
        # Test with sample texts
        test_texts = [
            "I love this! It's amazing!",
            "This is terrible and disappointing.",
            "The weather is nice today.",
            "I feel neutral about this topic."
        ]
        
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {text}")
            
            try:
                start_time = time.time()
                result = classifier(text)
                classification_time = time.time() - start_time
                
                print(f"Results ({classification_time:.2f}s):")
                # Show top 3 predictions
                if isinstance(result, list):
                    predictions = result[:3]  # Top 3
                    for pred in predictions:
                        print(f"  {pred['label']}: {pred['score']:.4f}")
                else:
                    print(f"  {result}")
                
            except Exception as e:
                print(f"✗ Classification failed: {str(e)}")
        
        # Clean up
        del classifier
        
        print(f"✓ {display_name} test completed")
        return True
        
    except Exception as e:
        print(f"✗ Failed to load {display_name}: {str(e)}")
        return False


def main():
    print("Testing Specific Hugging Face Models")
    print("====================================")
    
    results = {
        'text_generation': {},
        'text_classification': {}
    }
    
    # Text Generation Models
    print("\n🤖 TEXT GENERATION MODELS")
    generation_models = [
        ("gpt2", "GPT-2 Base"),
        ("microsoft/DialoGPT-medium", "DialoGPT Medium"),
        ("facebook/blenderbot_small-90M", "BlenderBot Small")
    ]
    
    for model_name, display_name in generation_models:
        success = test_text_generation_model(model_name, display_name)
        results['text_generation'][display_name] = success
    
    # Text Classification Models  
    print("\n\n📊 TEXT CLASSIFICATION MODELS")
    classification_models = [
        ("distilbert-base-uncased-finetuned-sst-2-english", "DistilBERT (BERT family)"),
        ("cardiffnlp/twitter-roberta-base-sentiment-latest", "RoBERTa Sentiment"),
        ("microsoft/deberta-v3-base", "DeBERTa v3 Base")
    ]
    
    for model_name, display_name in classification_models:
        success = test_classification_model(model_name, display_name)
        results['text_classification'][display_name] = success
    
    # Summary
    print("\n\n📈 SUMMARY")
    print("="*50)
    
    print("\nText Generation Models:")
    for model, success in results['text_generation'].items():
        status = "✓ Working" if success else "✗ Failed"
        print(f"  {model}: {status}")
    
    print("\nText Classification Models:")
    for model, success in results['text_classification'].items():
        status = "✓ Working" if success else "✗ Failed"
        print(f"  {model}: {status}")
    
    # Calculate success rates
    gen_success = sum(results['text_generation'].values())
    gen_total = len(results['text_generation'])
    class_success = sum(results['text_classification'].values())
    class_total = len(results['text_classification'])
    
    print(f"\nOverall Results:")
    print(f"  Text Generation: {gen_success}/{gen_total} models working")
    print(f"  Text Classification: {class_success}/{class_total} models working")
    
    if gen_success + class_success == gen_total + class_total:
        print("\n🎉 All models tested successfully!")
    else:
        print(f"\n⚠ {(gen_total + class_total) - (gen_success + class_success)} models had issues")
        print("Note: First runs require model downloads and may take longer")


if __name__ == "__main__":
    main()
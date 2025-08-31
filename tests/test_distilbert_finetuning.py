#!/usr/bin/env python3
"""
Test script for DistilBERT fine-tuning functionality
"""

import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.distilbert_model import DistilBERTFineTuner, DistilBERTModel
from src.utils.config import get_settings

def test_sample_dataset_creation():
    """Test creating a sample dataset"""
    print("=== Testing Sample Dataset Creation ===")
    
    fine_tuner = DistilBERTFineTuner(num_labels=2)
    sample_data = fine_tuner.create_sample_dataset(num_samples=20)
    
    print(f"Created dataset with {len(sample_data)} samples")
    print("Sample data:")
    print(sample_data.head())
    print(f"Label distribution:\n{sample_data['label'].value_counts()}")
    
    return sample_data

def test_dataset_preparation():
    """Test dataset preparation and tokenization"""
    print("\n=== Testing Dataset Preparation ===")
    
    fine_tuner = DistilBERTFineTuner(num_labels=2)
    sample_data = fine_tuner.create_sample_dataset(num_samples=50)
    
    train_dataset, val_dataset = fine_tuner.prepare_dataset(
        data=sample_data,
        test_size=0.3,
        random_state=42
    )
    
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Check a sample
    sample = train_dataset[0]
    print(f"Sample input shape: {sample['input_ids'].shape}")
    print(f"Sample attention mask shape: {sample['attention_mask'].shape}")
    print(f"Sample label: {sample['labels']}")
    
    return train_dataset, val_dataset

def test_model_initialization():
    """Test model and tokenizer initialization"""
    print("\n=== Testing Model Initialization ===")
    
    fine_tuner = DistilBERTFineTuner(num_labels=2)
    fine_tuner._load_model_and_tokenizer()
    
    print(f"Model type: {type(fine_tuner.model)}")
    print(f"Tokenizer type: {type(fine_tuner.tokenizer)}")
    print(f"Device: {fine_tuner.device}")
    print(f"Model config: {fine_tuner.model.config}")
    
    return fine_tuner

def test_training_args_setup():
    """Test training arguments setup"""
    print("\n=== Testing Training Arguments ===")
    
    fine_tuner = DistilBERTFineTuner(num_labels=2)
    training_args = fine_tuner.setup_training_args(
        output_dir="./test_output",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        learning_rate=5e-5
    )
    
    print(f"Output dir: {training_args.output_dir}")
    print(f"Epochs: {training_args.num_train_epochs}")
    print(f"Batch size: {training_args.per_device_train_batch_size}")
    print(f"Learning rate: {training_args.learning_rate}")
    
    return training_args

def test_quick_fine_tuning():
    """Test a quick fine-tuning run with minimal data"""
    print("\n=== Testing Quick Fine-Tuning ===")
    
    try:
        # Initialize fine-tuner
        fine_tuner = DistilBERTFineTuner(num_labels=2)
        
        # Create small sample dataset
        sample_data = fine_tuner.create_sample_dataset(num_samples=40)
        train_dataset, val_dataset = fine_tuner.prepare_dataset(
            data=sample_data,
            test_size=0.25,
            random_state=42
        )
        
        # Setup minimal training args
        training_args = fine_tuner.setup_training_args(
            output_dir="./test_fine_tuned_model",
            num_train_epochs=1,
            per_device_train_batch_size=4,
            learning_rate=5e-5,
            eval_steps=10,
            save_steps=50,
            logging_steps=5,
            warmup_steps=0
        )
        
        print("Starting quick fine-tuning...")
        results = fine_tuner.fine_tune(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            training_args=training_args,
            early_stopping_patience=1
        )
        
        print("Fine-tuning results:")
        print(f"  Training time: {results['training_time_seconds']:.2f} seconds")
        print(f"  Train loss: {results['train_loss']:.4f}")
        print(f"  Eval accuracy: {results['eval_metrics'].get('eval_accuracy', 'N/A'):.4f}")
        print(f"  Eval F1: {results['eval_metrics'].get('eval_f1', 'N/A'):.4f}")
        
        # Test prediction
        test_texts = [
            "This is absolutely amazing!",
            "I hate this terrible product."
        ]
        
        for text in test_texts:
            prediction = fine_tuner.predict(text, return_probabilities=True)
            print(f"\nText: {text}")
            print(f"Prediction: {prediction}")
        
        # Save the model
        fine_tuner.save_model("./test_fine_tuned_model")
        print("\nModel saved successfully")
        
        return results
        
    except Exception as e:
        print(f"Error during fine-tuning: {e}")
        return None

def test_model_loading():
    """Test loading a fine-tuned model"""
    print("\n=== Testing Model Loading ===")
    
    try:
        # Test if we have a saved model
        model_path = "./test_fine_tuned_model"
        if not os.path.exists(model_path):
            print("No fine-tuned model found, skipping loading test")
            return
        
        # Load the fine-tuned model
        model = DistilBERTModel(model_path=model_path)
        
        # Test predictions
        test_texts = [
            "Great product, highly recommended!",
            "Worst purchase ever, complete waste of money."
        ]
        
        for text in test_texts:
            result = model.classify_text(text, return_probabilities=True)
            print(f"Text: {text}")
            print(f"Classification: {result}")
            print()
        
        print("Model loading and prediction test successful!")
        
    except Exception as e:
        print(f"Error testing model loading: {e}")

def test_configuration():
    """Test configuration settings"""
    print("=== Testing Configuration ===")
    
    settings = get_settings()
    print(f"DistilBERT model name: {settings.distilbert_model_name}")
    print(f"Fine-tune output dir: {settings.fine_tune_output_dir}")
    print(f"Fine-tune batch size: {settings.fine_tune_batch_size}")
    print(f"Fine-tune learning rate: {settings.fine_tune_learning_rate}")
    print(f"Fine-tune epochs: {settings.fine_tune_epochs}")
    print(f"Use GPU: {settings.use_gpu}")
    print(f"CUDA available: {torch.cuda.is_available()}")

def main():
    """Run all tests"""
    print("DistilBERT Fine-Tuning Test Suite")
    print("=" * 50)
    
    # Test configuration
    test_configuration()
    
    # Test dataset creation
    sample_data = test_sample_dataset_creation()
    
    # Test dataset preparation
    train_dataset, val_dataset = test_dataset_preparation()
    
    # Test model initialization
    fine_tuner = test_model_initialization()
    
    # Test training arguments
    training_args = test_training_args_setup()
    
    # Test actual fine-tuning (optional, can be slow)
    print("\n" + "=" * 50)
    response = input("Run actual fine-tuning test? (y/N): ").strip().lower()
    if response == 'y':
        results = test_quick_fine_tuning()
        if results:
            test_model_loading()
    else:
        print("Skipping fine-tuning test")
    
    print("\n=== Test Suite Complete ===")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Example script demonstrating DistilBERT fine-tuning functionality
"""

from src.models.distilbert_model import DistilBERTFineTuner, DistilBERTModel
import pandas as pd

def example_fine_tuning():
    """Example of fine-tuning DistilBERT on sample data"""
    print("=== DistilBERT Fine-Tuning Example ===\n")
    
    # Initialize fine-tuner for binary classification
    fine_tuner = DistilBERTFineTuner(num_labels=2)
    print("✓ Fine-tuner initialized")
    
    # Create sample dataset
    sample_data = fine_tuner.create_sample_dataset(num_samples=100)
    print(f"✓ Created sample dataset with {len(sample_data)} samples")
    
    # Prepare training/validation split
    train_dataset, val_dataset = fine_tuner.prepare_dataset(
        data=sample_data,
        test_size=0.2,
        random_state=42
    )
    print(f"✓ Dataset split: {len(train_dataset)} train, {len(val_dataset)} validation")
    
    # Setup training arguments for quick training
    training_args = fine_tuner.setup_training_args(
        output_dir="./example_fine_tuned_model",
        num_train_epochs=1,  # Quick training for example
        per_device_train_batch_size=8,
        learning_rate=5e-5,
        eval_strategy="epoch",  # Evaluate once per epoch
        save_strategy="epoch",
        logging_steps=10,
        warmup_steps=0
    )
    print("✓ Training arguments configured")
    
    print("\nStarting fine-tuning (this may take a few minutes)...")
    
    # Run fine-tuning
    results = fine_tuner.fine_tune(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        training_args=training_args,
        early_stopping_patience=2
    )
    
    print("✓ Fine-tuning completed!")
    print(f"  Training time: {results['training_time_seconds']:.2f} seconds")
    print(f"  Final validation accuracy: {results['eval_metrics'].get('eval_accuracy', 'N/A'):.4f}")
    
    # Save the model
    fine_tuner.save_model("./example_fine_tuned_model")
    print("✓ Model saved to ./example_fine_tuned_model")
    
    # Test predictions with the fine-tuned model
    print("\n=== Testing Fine-Tuned Model Predictions ===")
    test_texts = [
        "This product is absolutely fantastic and I love it!",
        "Terrible quality, completely disappointed with this purchase.",
        "Great value for money, highly recommended!",
        "Worst experience ever, avoid at all costs."
    ]
    
    for text in test_texts:
        prediction = fine_tuner.predict(text, return_probabilities=True)
        print(f"\nText: {text}")
        print(f"Predicted class: {prediction.get('predicted_class', prediction['predicted_label'])}")
        print(f"Confidence: {prediction['confidence']:.4f}")
    
    return results

def example_using_fine_tuned_model():
    """Example of loading and using a fine-tuned model"""
    print("\n=== Using Fine-Tuned Model ===\n")
    
    model_path = "./example_fine_tuned_model"
    
    try:
        # Load fine-tuned model
        model = DistilBERTModel(model_path=model_path)
        print("✓ Fine-tuned model loaded")
        
        # Test classification
        test_texts = [
            "Amazing product, exceeded expectations!",
            "Poor quality, not worth the money."
        ]
        
        for text in test_texts:
            result = model.classify_text(text, return_probabilities=True)
            print(f"\nText: {text}")
            print(f"Classification: {result}")
            
    except Exception as e:
        print(f"Could not load fine-tuned model: {e}")
        print("Make sure to run the fine-tuning example first!")

def example_custom_dataset():
    """Example using custom dataset format"""
    print("\n=== Custom Dataset Example ===\n")
    
    # Create custom dataset
    custom_data = pd.DataFrame([
        {"text": "I love this new feature!", "label": "positive"},
        {"text": "This update is terrible.", "label": "negative"},
        {"text": "Great improvement to the app.", "label": "positive"},
        {"text": "Buggy and unreliable.", "label": "negative"},
        {"text": "Excellent user interface design.", "label": "positive"},
        {"text": "Confusing and hard to use.", "label": "negative"},
    ])
    
    print("Custom dataset:")
    print(custom_data)
    
    # Fine-tune with custom data
    fine_tuner = DistilBERTFineTuner(num_labels=2)
    train_dataset, val_dataset = fine_tuner.prepare_dataset(
        data=custom_data,
        text_column="text",
        label_column="label",
        test_size=0.3
    )
    
    print(f"\n✓ Custom dataset prepared: {len(train_dataset)} train, {len(val_dataset)} val")
    print(f"Label mapping: {fine_tuner.label_mapping}")

def main():
    """Run all examples"""
    try:
        # Check if we want to run the actual fine-tuning
        print("This example demonstrates DistilBERT fine-tuning capabilities.")
        print("Note: Actual fine-tuning may take several minutes.\n")
        
        # Always run the custom dataset example (quick)
        example_custom_dataset()
        
        # Ask about running full fine-tuning
        response = input("\nRun full fine-tuning example? (y/N): ").strip().lower()
        if response == 'y':
            results = example_fine_tuning()
            example_using_fine_tuned_model()
        else:
            print("\nSkipping fine-tuning example.")
            print("To run fine-tuning manually:")
            print("1. Initialize: fine_tuner = DistilBERTFineTuner(num_labels=2)")
            print("2. Prepare data: train_ds, val_ds = fine_tuner.prepare_dataset(your_data)")
            print("3. Fine-tune: results = fine_tuner.fine_tune(train_ds, val_ds)")
        
        print("\n✓ DistilBERT fine-tuning implementation ready for use!")
        
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"\nError running example: {e}")

if __name__ == "__main__":
    main()
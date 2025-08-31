"""
DistilBERT model with fine-tuning capabilities for text classification and generation
"""

import os
import json
import torch
from torch.utils.data import Dataset
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    pipeline
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from functools import lru_cache

from src.utils.config import get_settings, create_model_cache_dir


class TextClassificationDataset(Dataset):
    """Custom dataset for text classification fine-tuning"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


class DistilBERTFineTuner:
    """
    DistilBERT fine-tuning implementation for text classification
    """
    
    def __init__(self, model_name: str = None, num_labels: int = 2):
        self.settings = get_settings()
        self.model_name = model_name or self.settings.distilbert_model_name
        self.cache_dir = create_model_cache_dir()
        self.use_gpu = self.settings.use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        self.num_labels = num_labels
        
        # Initialize logging
        log_level = getattr(logging, self.settings.log_level.upper(), logging.INFO)
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
        # Training history
        self.training_history = []
        
        self.logger.info(f"DistilBERT fine-tuner initialized")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Number of labels: {self.num_labels}")
        
    def _load_model_and_tokenizer(self):
        """Load the tokenizer and model"""
        try:
            self.logger.info("Loading tokenizer and model...")
            
            # Load tokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir
            )
            
            # Load model for classification
            self.model = DistilBertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=self.num_labels,
                cache_dir=self.cache_dir
            )
            
            # Move to device
            self.model = self.model.to(self.device)
            
            self.logger.info("Model and tokenizer loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading model and tokenizer: {e}")
            raise
    
    def prepare_dataset(
        self, 
        data: Union[pd.DataFrame, List[Dict[str, Any]]], 
        text_column: str = "text", 
        label_column: str = "label",
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[Dataset, Dataset]:
        """
        Prepare training and validation datasets
        
        Args:
            data: DataFrame or list of dictionaries with text and labels
            text_column: Name of the text column
            label_column: Name of the label column
            test_size: Proportion of data to use for validation
            random_state: Random state for reproducible splits
            
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        if not self.tokenizer:
            self._load_model_and_tokenizer()
        
        # Convert to DataFrame if needed
        if isinstance(data, list):
            data = pd.DataFrame(data)
        
        # Extract texts and labels
        texts = data[text_column].tolist()
        labels = data[label_column].tolist()
        
        # Create label mapping if labels are strings
        if isinstance(labels[0], str):
            unique_labels = list(set(labels))
            label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
            labels = [label_to_id[label] for label in labels]
            
            # Save label mapping for later use
            self.label_mapping = {v: k for k, v in label_to_id.items()}
            self.logger.info(f"Created label mapping: {label_to_id}")
        else:
            self.label_mapping = {i: str(i) for i in range(self.num_labels)}
        
        # Split data
        from sklearn.model_selection import train_test_split
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=test_size, random_state=random_state, stratify=labels
        )
        
        # Create datasets
        train_dataset = TextClassificationDataset(
            train_texts, train_labels, self.tokenizer, max_length=512
        )
        val_dataset = TextClassificationDataset(
            val_texts, val_labels, self.tokenizer, max_length=512
        )
        
        self.logger.info(f"Dataset prepared: {len(train_dataset)} training, {len(val_dataset)} validation samples")
        
        return train_dataset, val_dataset
    
    def setup_training_args(
        self,
        output_dir: str = "./results",
        num_train_epochs: int = 3,
        per_device_train_batch_size: int = 16,
        per_device_eval_batch_size: int = 16,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_steps: int = 500,
        eval_strategy: str = "steps",
        eval_steps: int = 500,
        save_strategy: str = "steps",
        save_steps: int = 500,
        logging_steps: int = 100,
        load_best_model_at_end: bool = True,
        metric_for_best_model: str = "eval_f1",
        greater_is_better: bool = True,
        **kwargs
    ) -> TrainingArguments:
        """
        Setup training arguments with sensible defaults
        
        Returns:
            TrainingArguments object
        """
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_eval_batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            save_strategy=save_strategy,
            save_steps=save_steps,
            logging_steps=logging_steps,
            load_best_model_at_end=load_best_model_at_end,
            metric_for_best_model=metric_for_best_model,
            greater_is_better=greater_is_better,
            remove_unused_columns=False,
            push_to_hub=False,
            **kwargs
        )
        
        return training_args
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def fine_tune(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        training_args: Optional[TrainingArguments] = None,
        early_stopping_patience: int = 3
    ) -> Dict[str, Any]:
        """
        Fine-tune the DistilBERT model
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            training_args: Training arguments (if None, uses defaults)
            early_stopping_patience: Early stopping patience
            
        Returns:
            Dictionary with training results and metrics
        """
        if not self.model:
            self._load_model_and_tokenizer()
        
        # Use default training args if none provided
        if training_args is None:
            training_args = self.setup_training_args()
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)]
        )
        
        # Start training
        self.logger.info("Starting fine-tuning...")
        start_time = datetime.now()
        
        try:
            train_result = self.trainer.train()
            
            # Calculate training time
            end_time = datetime.now()
            training_time = (end_time - start_time).total_seconds()
            
            # Evaluate on validation set
            eval_result = self.trainer.evaluate()
            
            # Prepare results
            results = {
                'training_time_seconds': training_time,
                'train_loss': train_result.training_loss,
                'eval_metrics': eval_result,
                'best_model_checkpoint': self.trainer.state.best_model_checkpoint,
                'global_step': train_result.global_step,
                'training_completed': True,
                'timestamp': datetime.now().isoformat()
            }
            
            # Store in training history
            self.training_history.append(results)
            
            self.logger.info(f"Fine-tuning completed in {training_time:.2f} seconds")
            self.logger.info(f"Final validation accuracy: {eval_result.get('eval_accuracy', 'N/A'):.4f}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {e}")
            error_result = {
                'training_completed': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.training_history.append(error_result)
            raise
    
    def save_model(self, save_path: str, save_tokenizer: bool = True):
        """Save the fine-tuned model and tokenizer"""
        if not self.model:
            raise ValueError("No model to save. Train the model first.")
        
        os.makedirs(save_path, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        
        # Save tokenizer
        if save_tokenizer and self.tokenizer:
            self.tokenizer.save_pretrained(save_path)
        
        # Save label mapping
        if hasattr(self, 'label_mapping'):
            label_mapping_path = os.path.join(save_path, 'label_mapping.json')
            with open(label_mapping_path, 'w') as f:
                json.dump(self.label_mapping, f, indent=2)
        
        # Save training history
        history_path = os.path.join(save_path, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        self.logger.info(f"Model saved to {save_path}")
    
    def load_fine_tuned_model(self, model_path: str):
        """Load a fine-tuned model"""
        self.logger.info(f"Loading fine-tuned model from {model_path}")
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        
        # Load model
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model = self.model.to(self.device)
        
        # Load label mapping if available
        label_mapping_path = os.path.join(model_path, 'label_mapping.json')
        if os.path.exists(label_mapping_path):
            with open(label_mapping_path, 'r') as f:
                self.label_mapping = json.load(f)
        
        # Load training history if available
        history_path = os.path.join(model_path, 'training_history.json')
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
        
        self.logger.info("Fine-tuned model loaded successfully")
    
    def predict(self, texts: Union[str, List[str]], return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Make predictions using the fine-tuned model
        
        Args:
            texts: Single text or list of texts to classify
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dictionary with predictions and optionally probabilities
        """
        if not self.model:
            raise ValueError("No model loaded. Load a model first.")
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
            single_text = True
        else:
            single_text = False
        
        # Create pipeline for inference
        classifier = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if self.use_gpu else -1,
            return_all_scores=return_probabilities
        )
        
        # Make predictions
        predictions = classifier(texts)
        
        # Process results
        results = []
        for i, pred in enumerate(predictions):
            if return_probabilities:
                # Sort by score and include all probabilities
                sorted_preds = sorted(pred, key=lambda x: x['score'], reverse=True)
                result = {
                    'text': texts[i],
                    'predicted_label': sorted_preds[0]['label'],
                    'confidence': sorted_preds[0]['score'],
                    'all_probabilities': {p['label']: p['score'] for p in sorted_preds}
                }
            else:
                result = {
                    'text': texts[i],
                    'predicted_label': pred['label'],
                    'confidence': pred['score']
                }
            
            # Map back to original labels if mapping exists
            if hasattr(self, 'label_mapping') and result['predicted_label'].startswith('LABEL_'):
                label_idx = int(result['predicted_label'].split('_')[1])
                if str(label_idx) in self.label_mapping:
                    result['predicted_class'] = self.label_mapping[str(label_idx)]
            
            results.append(result)
        
        # Return single result if input was single text
        if single_text:
            return results[0]
        
        return {'predictions': results}
    
    def evaluate_model(self, test_dataset: Dataset) -> Dict[str, Any]:
        """
        Evaluate the model on a test dataset
        
        Args:
            test_dataset: Test dataset
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.trainer:
            raise ValueError("No trainer available. Train the model first.")
        
        self.logger.info("Evaluating model...")
        
        # Run evaluation
        eval_result = self.trainer.evaluate(eval_dataset=test_dataset)
        
        # Generate detailed classification report if possible
        try:
            # Get predictions for detailed report
            predictions = self.trainer.predict(test_dataset)
            y_pred = np.argmax(predictions.predictions, axis=1)
            y_true = predictions.label_ids
            
            # Generate classification report
            if hasattr(self, 'label_mapping'):
                target_names = [self.label_mapping.get(str(i), f"Class_{i}") for i in range(self.num_labels)]
            else:
                target_names = [f"Class_{i}" for i in range(self.num_labels)]
            
            classification_rep = classification_report(
                y_true, y_pred, target_names=target_names, output_dict=True
            )
            
            eval_result['detailed_classification_report'] = classification_rep
            
        except Exception as e:
            self.logger.warning(f"Could not generate detailed classification report: {e}")
        
        self.logger.info(f"Evaluation completed. Accuracy: {eval_result.get('eval_accuracy', 'N/A'):.4f}")
        
        return eval_result
    
    def load_data_from_file(self, file_path: str, format: str = "auto") -> pd.DataFrame:
        """
        Load training data from various file formats
        
        Args:
            file_path: Path to the data file
            format: File format ("csv", "json", "jsonl", "auto")
            
        Returns:
            DataFrame with the loaded data
        """
        if format == "auto":
            format = file_path.split('.')[-1].lower()
        
        self.logger.info(f"Loading data from {file_path} (format: {format})")
        
        if format == "csv":
            data = pd.read_csv(file_path)
        elif format == "json":
            data = pd.read_json(file_path)
        elif format == "jsonl":
            data = pd.read_json(file_path, lines=True)
        else:
            raise ValueError(f"Unsupported file format: {format}")
        
        self.logger.info(f"Loaded {len(data)} samples")
        return data
    
    def create_sample_dataset(self, num_samples: int = 100) -> pd.DataFrame:
        """
        Create a sample dataset for testing fine-tuning
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            DataFrame with sample data
        """
        import random
        
        # Sample texts and labels for binary sentiment classification
        positive_texts = [
            "This is excellent and amazing!",
            "I love this product, it's fantastic.",
            "Great quality and fast delivery.",
            "Highly recommended, very satisfied.",
            "Outstanding service and support.",
            "Perfect solution for my needs.",
            "Impressive results and easy to use.",
            "Exceeded my expectations completely."
        ]
        
        negative_texts = [
            "This is terrible and disappointing.",
            "Poor quality, waste of money.",
            "Horrible experience, would not recommend.",
            "Completely useless and frustrating.",
            "Terrible customer service.",
            "Product broke immediately.",
            "Very poor performance and reliability.",
            "Disappointed with the results."
        ]
        
        # Generate samples
        data = []
        for i in range(num_samples):
            if i % 2 == 0:
                text = random.choice(positive_texts)
                label = 1  # positive
            else:
                text = random.choice(negative_texts)
                label = 0  # negative
            
            data.append({'text': text, 'label': label})
        
        df = pd.DataFrame(data)
        self.logger.info(f"Created sample dataset with {len(df)} samples")
        
        return df


class DistilBERTModel:
    """
    Main DistilBERT model class for inference with optional fine-tuning support
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.settings = get_settings()
        self.cache_dir = create_model_cache_dir()
        self.use_gpu = self.settings.use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # Initialize logging
        log_level = getattr(logging, self.settings.log_level.upper(), logging.INFO)
        logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Model components
        self.pipeline = None
        self.fine_tuner = None
        
        # Load model
        if model_path:
            self._load_fine_tuned_model(model_path)
        else:
            self._initialize_default_pipeline()
        
        self.logger.info("DistilBERT model initialized")
    
    def _initialize_default_pipeline(self):
        """Initialize default DistilBERT pipeline for classification"""
        try:
            self.logger.info("Loading default DistilBERT pipeline...")
            
            self.pipeline = pipeline(
                "text-classification",
                model=self.settings.distilbert_model_name,
                cache_dir=self.cache_dir,
                device=0 if self.use_gpu else -1
            )
            
            self.logger.info("Default DistilBERT pipeline loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error loading default pipeline: {e}")
            raise
    
    def _load_fine_tuned_model(self, model_path: str):
        """Load a fine-tuned model"""
        self.fine_tuner = DistilBERTFineTuner()
        self.fine_tuner.load_fine_tuned_model(model_path)
        
        # Create pipeline from fine-tuned model
        self.pipeline = pipeline(
            "text-classification",
            model=self.fine_tuner.model,
            tokenizer=self.fine_tuner.tokenizer,
            device=0 if self.use_gpu else -1
        )
    
    def classify_text(self, text: str, return_probabilities: bool = False) -> Dict[str, Any]:
        """
        Classify text using the loaded model
        
        Args:
            text: Input text to classify
            return_probabilities: Whether to return all class probabilities
            
        Returns:
            Dictionary with classification results
        """
        if not self.pipeline:
            raise ValueError("No model pipeline available")
        
        try:
            # Use fine-tuned model if available
            if self.fine_tuner:
                return self.fine_tuner.predict(text, return_probabilities=return_probabilities)
            
            # Use default pipeline
            result = self.pipeline(text, return_all_scores=return_probabilities)
            
            if return_probabilities:
                return {
                    'text': text,
                    'predicted_label': result[0]['label'],
                    'confidence': result[0]['score'],
                    'all_probabilities': {r['label']: r['score'] for r in result}
                }
            else:
                return {
                    'text': text,
                    'predicted_label': result['label'],
                    'confidence': result['score']
                }
        
        except Exception as e:
            self.logger.error(f"Error in text classification: {e}")
            raise
    
    def get_fine_tuner(self, num_labels: int = 2) -> DistilBERTFineTuner:
        """
        Get a fine-tuner instance for this model
        
        Args:
            num_labels: Number of classification labels
            
        Returns:
            DistilBERTFineTuner instance
        """
        return DistilBERTFineTuner(model_name=self.settings.distilbert_model_name, num_labels=num_labels)


@lru_cache()
def get_distilbert_model(model_path: Optional[str] = None) -> DistilBERTModel:
    """Get cached DistilBERT model instance"""
    return DistilBERTModel(model_path=model_path)


# Utility functions for easy access
def quick_classify(text: str, model_path: Optional[str] = None) -> Dict[str, Any]:
    """Quick text classification"""
    model = get_distilbert_model(model_path)
    return model.classify_text(text)


def create_fine_tuning_session(num_labels: int = 2) -> DistilBERTFineTuner:
    """Create a new fine-tuning session"""
    return DistilBERTFineTuner(num_labels=num_labels)
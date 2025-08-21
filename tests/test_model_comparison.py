#!/usr/bin/env python3
"""
Comprehensive test script for comparing various Hugging Face models
Tests text generation and classification models with performance metrics
"""
import os
import sys
import time
import psutil
import json
from typing import Dict, List, Any, Tuple
from datetime import datetime
from dotenv import load_dotenv
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

load_dotenv()

class ModelPerformanceTester:
    """Test and compare performance of different Hugging Face models"""
    
    def __init__(self):
        self.results = {
            'text_generation': {},
            'text_classification': {},
            'test_metadata': {
                'timestamp': datetime.now().isoformat(),
                'system_info': self._get_system_info()
            }
        }
        
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information for context"""
        return {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'python_version': sys.version.split()[0]
        }
    
    def _measure_performance(self, func, *args, **kwargs) -> Tuple[Any, Dict[str, Any]]:
        """Measure performance metrics for a function call"""
        process = psutil.Process()
        
        # Initial measurements
        start_time = time.time()
        start_memory = process.memory_info().rss / (1024**2)  # MB
        start_cpu_percent = process.cpu_percent()
        
        try:
            # Execute function
            result = func(*args, **kwargs)
            
            # Final measurements
            end_time = time.time()
            end_memory = process.memory_info().rss / (1024**2)  # MB
            end_cpu_percent = process.cpu_percent()
            
            metrics = {
                'execution_time': round(end_time - start_time, 3),
                'memory_used_mb': round(end_memory - start_memory, 2),
                'peak_memory_mb': round(end_memory, 2),
                'cpu_usage_change': round(end_cpu_percent - start_cpu_percent, 2),
                'success': True,
                'error': None
            }
            
            return result, metrics
            
        except Exception as e:
            end_time = time.time()
            metrics = {
                'execution_time': round(end_time - start_time, 3),
                'memory_used_mb': 0,
                'peak_memory_mb': 0,
                'cpu_usage_change': 0,
                'success': False,
                'error': str(e)
            }
            return None, metrics
    
    def test_text_generation_models(self) -> None:
        """Test various text generation models"""
        print("=== Testing Text Generation Models ===\n")
        
        # Models to test
        models_to_test = [
            ("gpt2", "GPT-2 Base"),
            ("facebook/blenderbot_small-90M", "BlenderBot Small"),
        ]
        
        test_prompts = [
            "Hello, how are you today?",
            "Tell me about artificial intelligence.",
            "What's the weather like?",
            "Can you help me with programming?",
            "Explain quantum computing briefly."
        ]
        
        for model_name, display_name in models_to_test:
            print(f"Testing {display_name} ({model_name})...")
            
            try:
                # Import here to avoid loading all models at startup
                from src.models.huggingface_client import HuggingFaceClient
                from src.utils.config import get_settings
                
                # Temporarily override model name
                settings = get_settings()
                original_model = settings.huggingface_model_name
                settings.huggingface_model_name = model_name
                
                # Initialize client with new model
                client = HuggingFaceClient()
                
                model_results = {
                    'model_info': client.get_model_info(),
                    'prompts': {},
                    'averages': {}
                }
                
                total_time = 0
                total_memory = 0
                successful_tests = 0
                
                for i, prompt in enumerate(test_prompts):
                    print(f"  Prompt {i+1}/5: {prompt[:30]}...")
                    
                    # Test generation
                    result, metrics = self._measure_performance(
                        client.generate_text,
                        prompt,
                        max_length=50
                    )
                    
                    if metrics['success']:
                        prompt_result = {
                            'prompt': prompt,
                            'generated_text': result['text'][:100] + '...' if len(result['text']) > 100 else result['text'],
                            'token_usage': result['usage'],
                            'performance': metrics
                        }
                        successful_tests += 1
                        total_time += metrics['execution_time']
                        total_memory += metrics['memory_used_mb']
                    else:
                        prompt_result = {
                            'prompt': prompt,
                            'error': metrics['error'],
                            'performance': metrics
                        }
                    
                    model_results['prompts'][f'prompt_{i+1}'] = prompt_result
                
                # Calculate averages
                if successful_tests > 0:
                    model_results['averages'] = {
                        'avg_execution_time': round(total_time / successful_tests, 3),
                        'avg_memory_usage': round(total_memory / successful_tests, 2),
                        'success_rate': round(successful_tests / len(test_prompts) * 100, 1)
                    }
                
                self.results['text_generation'][model_name] = model_results
                
                # Clean up
                client.unload_model()
                del client
                
                # Restore original model
                settings.huggingface_model_name = original_model
                
                print(f"  ✓ Completed {display_name}")
                if successful_tests > 0:
                    print(f"    Success rate: {model_results['averages']['success_rate']}%")
                    print(f"    Avg time: {model_results['averages']['avg_execution_time']}s")
                    print(f"    Avg memory: {model_results['averages']['avg_memory_usage']}MB\n")
                else:
                    print("    ✗ All tests failed\n")
                
            except Exception as e:
                print(f"  ✗ Failed to test {display_name}: {str(e)}\n")
                self.results['text_generation'][model_name] = {
                    'error': str(e),
                    'model_info': None,
                    'prompts': {},
                    'averages': {}
                }
    
    def test_text_classification_models(self) -> None:
        """Test various text classification models"""
        print("=== Testing Text Classification Models ===\n")
        
        # Models to test
        models_to_test = [
            ("distilbert-base-uncased-finetuned-sst-2-english", "DistilBERT SST-2"),
            ("cardiffnlp/twitter-roberta-base-sentiment-latest", "RoBERTa Sentiment"),
        ]
        
        test_texts = [
            "I love this new feature! It's amazing!",
            "This is terrible. I hate it.",
            "The weather is nice today.",
            "I'm feeling quite neutral about this.",
            "This is the best thing that ever happened to me!",
            "I'm not sure how I feel about this situation.",
            "The product quality is disappointing.",
            "Everything is going perfectly as planned."
        ]
        
        for model_name, display_name in models_to_test:
            print(f"Testing {display_name} ({model_name})...")
            
            try:
                from transformers import pipeline
                
                # Initialize pipeline
                classifier, init_metrics = self._measure_performance(
                    pipeline,
                    "text-classification",
                    model=model_name,
                    return_all_scores=True,
                    device=-1  # Force CPU for consistency
                )
                
                if not init_metrics['success']:
                    print(f"  ✗ Failed to load model: {init_metrics['error']}\n")
                    continue
                
                model_results = {
                    'model_name': model_name,
                    'initialization_time': init_metrics['execution_time'],
                    'initialization_memory': init_metrics['memory_used_mb'],
                    'texts': {},
                    'averages': {}
                }
                
                total_time = 0
                total_memory = 0
                successful_tests = 0
                
                for i, text in enumerate(test_texts):
                    print(f"  Text {i+1}/8: {text[:30]}...")
                    
                    # Test classification
                    result, metrics = self._measure_performance(
                        classifier,
                        text
                    )
                    
                    if metrics['success']:
                        text_result = {
                            'text': text,
                            'predictions': result[:3] if isinstance(result, list) else result,  # Top 3
                            'performance': metrics
                        }
                        successful_tests += 1
                        total_time += metrics['execution_time']
                        total_memory += metrics['memory_used_mb']
                    else:
                        text_result = {
                            'text': text,
                            'error': metrics['error'],
                            'performance': metrics
                        }
                    
                    model_results['texts'][f'text_{i+1}'] = text_result
                
                # Calculate averages
                if successful_tests > 0:
                    model_results['averages'] = {
                        'avg_execution_time': round(total_time / successful_tests, 3),
                        'avg_memory_usage': round(total_memory / successful_tests, 2),
                        'success_rate': round(successful_tests / len(test_texts) * 100, 1)
                    }
                
                self.results['text_classification'][model_name] = model_results
                
                # Clean up
                del classifier
                
                print(f"  ✓ Completed {display_name}")
                if successful_tests > 0:
                    print(f"    Success rate: {model_results['averages']['success_rate']}%")
                    print(f"    Avg time: {model_results['averages']['avg_execution_time']}s")
                    print(f"    Avg memory: {model_results['averages']['avg_memory_usage']}MB\n")
                else:
                    print("    ✗ All tests failed\n")
                
            except Exception as e:
                print(f"  ✗ Failed to test {display_name}: {str(e)}\n")
                self.results['text_classification'][model_name] = {
                    'error': str(e),
                    'model_name': model_name,
                    'texts': {},
                    'averages': {}
                }
    
    def generate_comparison_report(self) -> str:
        """Generate a comprehensive comparison report"""
        report = []
        report.append("# Hugging Face Model Comparison Report")
        report.append(f"Generated on: {self.results['test_metadata']['timestamp']}")
        report.append(f"System: {self.results['test_metadata']['system_info']}")
        report.append("")
        
        # Text Generation Results
        report.append("## Text Generation Models")
        report.append("")
        
        gen_models = self.results['text_generation']
        if gen_models:
            # Create comparison table
            report.append("| Model | Success Rate | Avg Time (s) | Avg Memory (MB) | Notes |")
            report.append("|-------|-------------|-------------|----------------|-------|")
            
            for model_name, data in gen_models.items():
                if 'averages' in data and data['averages']:
                    avg = data['averages']
                    success_rate = avg.get('success_rate', 0)
                    avg_time = avg.get('avg_execution_time', 'N/A')
                    avg_memory = avg.get('avg_memory_usage', 'N/A')
                    notes = "✓ Working" if success_rate > 80 else "⚠ Issues"
                else:
                    success_rate = 0
                    avg_time = 'N/A'
                    avg_memory = 'N/A'
                    notes = "✗ Failed"
                
                report.append(f"| {model_name} | {success_rate}% | {avg_time} | {avg_memory} | {notes} |")
        
        report.append("")
        
        # Text Classification Results
        report.append("## Text Classification Models")
        report.append("")
        
        class_models = self.results['text_classification']
        if class_models:
            # Create comparison table
            report.append("| Model | Success Rate | Avg Time (s) | Avg Memory (MB) | Init Time (s) | Notes |")
            report.append("|-------|-------------|-------------|----------------|---------------|-------|")
            
            for model_name, data in class_models.items():
                if 'averages' in data and data['averages']:
                    avg = data['averages']
                    success_rate = avg.get('success_rate', 0)
                    avg_time = avg.get('avg_execution_time', 'N/A')
                    avg_memory = avg.get('avg_memory_usage', 'N/A')
                    init_time = data.get('initialization_time', 'N/A')
                    notes = "✓ Working" if success_rate > 80 else "⚠ Issues"
                else:
                    success_rate = 0
                    avg_time = 'N/A'
                    avg_memory = 'N/A'
                    init_time = 'N/A'
                    notes = "✗ Failed"
                
                report.append(f"| {model_name} | {success_rate}% | {avg_time} | {avg_memory} | {init_time} | {notes} |")
        
        report.append("")
        report.append("## Recommendations")
        report.append("")
        
        # Add recommendations based on results
        if gen_models:
            best_gen_model = None
            best_score = -1
            
            for model_name, data in gen_models.items():
                if 'averages' in data and data['averages']:
                    # Score based on success rate and speed (higher is better)
                    success_rate = data['averages'].get('success_rate', 0)
                    avg_time = data['averages'].get('avg_execution_time', float('inf'))
                    score = success_rate / max(avg_time, 0.1)  # Avoid division by zero
                    
                    if score > best_score:
                        best_score = score
                        best_gen_model = model_name
            
            if best_gen_model:
                report.append(f"**Best Text Generation Model:** {best_gen_model}")
                report.append("")
        
        if class_models:
            best_class_model = None
            best_score = -1
            
            for model_name, data in class_models.items():
                if 'averages' in data and data['averages']:
                    # Score based on success rate and speed
                    success_rate = data['averages'].get('success_rate', 0)
                    avg_time = data['averages'].get('avg_execution_time', float('inf'))
                    score = success_rate / max(avg_time, 0.1)
                    
                    if score > best_score:
                        best_score = score
                        best_class_model = model_name
            
            if best_class_model:
                report.append(f"**Best Text Classification Model:** {best_class_model}")
                report.append("")
        
        return "\n".join(report)
    
    def save_results(self, filename: str = None) -> str:
        """Save test results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"model_comparison_results_{timestamp}.json"
        
        filepath = os.path.join(project_root, "tests/reports", filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        return filepath


def main():
    """Run the comprehensive model comparison test"""
    print("Starting comprehensive Hugging Face model comparison...\n")
    print("⚠ Warning: This test will download multiple models and may take significant time!")
    print("⚠ Warning: Ensure you have sufficient disk space (several GB) and memory.\n")
    
    # Ask for confirmation
    try:
        confirm = input("Do you want to continue? (y/N): ").lower().strip()
        if confirm not in ['y', 'yes']:
            print("Test cancelled.")
            return
    except KeyboardInterrupt:
        print("\nTest cancelled.")
        return
    
    tester = ModelPerformanceTester()
    
    try:
        # Test text generation models
        tester.test_text_generation_models()
        
        # Test text classification models  
        tester.test_text_classification_models()
        
        # Generate and save results
        print("=== Generating Report ===")
        report = tester.generate_comparison_report()
        
        # Save JSON results
        results_file = tester.save_results()
        print(f"✓ Detailed results saved to: {results_file}")
        
        # Save markdown report
        report_file = results_file.replace('.json', '_report.md')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✓ Report saved to: {report_file}")
        
        # Display summary
        print("\n" + "="*50)
        print(report)
        print("="*50)
        
        print(f"\n🎉 Model comparison completed!")
        print(f"📊 Check {report_file} for the full report")
        print(f"📁 Raw data available in {results_file}")
        
    except KeyboardInterrupt:
        print("\n\n⚠ Test interrupted by user")
        print("Partial results may have been saved.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        print("Check the error message and try again.")


if __name__ == "__main__":
    main()
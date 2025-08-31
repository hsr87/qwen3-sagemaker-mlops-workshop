#!/usr/bin/env python3
import os
import json
import torch
import tarfile
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class TestDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=256):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item.get("text", "")
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "text": text
        }

def evaluate_model():
    # Paths
    model_path = "/opt/ml/processing/model"
    test_data_path = "/opt/ml/processing/input/test/test.jsonl"
    output_path = "/opt/ml/processing/output"
    
    # Extract model.tar.gz
    print("Extracting model artifacts...")
    with tarfile.open(os.path.join(model_path, "model.tar.gz"), "r:gz") as tar:
        tar.extractall(model_path)
    
    # Load tokenizer and base model - CPU optimized
    print("Loading tokenizer and base model...")
    model_name = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model for CPU
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,  # Changed to float32 for CPU
        device_map="cpu",  # Force CPU usage
        trust_remote_code=True
    )
    
    # Load LoRA adapter
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # Create test dataset
    print("Loading test data...")
    test_dataset = TestDataset(test_data_path, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    # Evaluation - CPU based
    print("Starting evaluation...")
    all_predictions = []
    all_references = []
    total_loss = 0
    num_samples = 0
    
    with torch.no_grad(): # cpu test setting
        for batch in tqdm(test_loader, desc="Evaluating"):
            input_ids = batch["input_ids"]  # No .cuda() for CPU
            attention_mask = batch["attention_mask"]  # No .cuda() for CPU
            
            # Generate predictions with reduced parameters for CPU
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,  # Reduced for CPU performance
                temperature=0.7,
                do_sample=True,
                top_p=0.95,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
            # Decode predictions
            pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            original_text = batch["text"][0]
            
            # Extract generated part (simple approach)
            if len(pred_text) > len(original_text):
                generated_text = pred_text[len(original_text):].strip()
            else:
                generated_text = pred_text
            
            all_predictions.append(generated_text)
            all_references.append(original_text)
            
            # Calculate perplexity instead of loss for CPU efficiency
            num_samples += 1
    
    # Calculate simple metrics
    avg_pred_length = np.mean([len(pred.split()) for pred in all_predictions])
    avg_ref_length = np.mean([len(ref.split()) for ref in all_references])
    
    # Simple text similarity metric - matching 2_evaluation.ipynb approach
    def calculate_similarity(predictions, references):
        similarities = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            if len(ref_words) == 0:
                similarities.append(0)
                continue
            
            intersection = pred_words & ref_words
            similarity = len(intersection) / len(ref_words) if len(ref_words) > 0 else 0
            similarities.append(similarity)
        
        return np.mean(similarities)
    
    text_similarity = calculate_similarity(all_predictions, all_references)
    
    # Prepare results - matching 2_evaluation.ipynb format with text_similarity as primary metric
    metrics = {
        "text_similarity": text_similarity,
        "avg_prediction_length": avg_pred_length,
        "avg_reference_length": avg_ref_length,
        "num_samples": num_samples,
        "model_location": model_path,
        "evaluation_type": "cpu_based"
    }
    
    # Save metrics
    print(f"\nEvaluation Results:")
    print(f"Text Similarity: {text_similarity:.4f}")
    print(f"Average Prediction Length: {avg_pred_length:.2f} words")
    print(f"Average Reference Length: {avg_ref_length:.2f} words")
    print(f"Number of samples: {num_samples}")
    
    # Save results to JSON
    with open(os.path.join(output_path, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Save sample predictions
    samples = []
    for i in range(min(5, len(all_predictions))):
        samples.append({
            "prediction": all_predictions[i],
            "reference": all_references[i]
        })
    
    with open(os.path.join(output_path, "sample_predictions.json"), "w") as f:
        json.dump(samples, f, indent=2, ensure_ascii=False)
    
    print("\nEvaluation completed successfully!")
    return metrics

if __name__ == "__main__":
    evaluate_model()

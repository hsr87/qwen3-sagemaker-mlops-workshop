#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def model_fn(model_dir):
    """
    Load the model for inference
    """
    print(f"Loading model from {model_dir}")
    
    # Model name from environment or default
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Check if LoRA adapter exists
    adapter_path = os.path.join(model_dir, "adapter_model.safetensors")
    if os.path.exists(adapter_path):
        print("Loading LoRA adapter")
        model = PeftModel.from_pretrained(model, model_dir)
        model = model.merge_and_unload()
    
    model.eval()
    
    return {"model": model, "tokenizer": tokenizer}


def input_fn(input_data, content_type):
    """
    Parse input data
    """
    if content_type == "application/json":
        data = json.loads(input_data)
        return data
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(data, model_dict):
    """
    Run inference
    """
    model = model_dict["model"]
    tokenizer = model_dict["tokenizer"]
    
    # Get input text and parameters
    if isinstance(data, dict):
        text = data.get("inputs", "")
        parameters = data.get("parameters", {})
    else:
        text = str(data)
        parameters = {}
    
    # Default parameters
    max_new_tokens = parameters.get("max_new_tokens", 50)
    temperature = parameters.get("temperature", 0.7)
    do_sample = parameters.get("do_sample", True)
    top_p = parameters.get("top_p", 0.9)
    
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode output
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Remove input text from output if it's included
    if generated_text.startswith(text):
        generated_text = generated_text[len(text):].strip()
    
    return {"generated_text": generated_text}


def output_fn(prediction, accept):
    """
    Format prediction output
    """
    if accept == "application/json":
        return json.dumps(prediction)
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
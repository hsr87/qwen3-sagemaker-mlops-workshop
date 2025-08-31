#!/usr/bin/env python3
"""
QWEN3-0.6B LoRA Fine-tuning on SageMaker
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

import torch
print(f"Using PyTorch version: {torch.__version__}")

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from datasets import load_dataset, DatasetDict
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

# Logging setup
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """Model related arguments"""
    model_name_or_path: str = field(
        default="Qwen/Qwen3-0.6B",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Will use the token generated when running `transformers-cli login`"}
    )


@dataclass
class DataArguments:
    """Data related arguments"""
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of training examples."}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes, truncate the number of evaluation examples."}
    )
    block_size: int = field(
        default=256,
        metadata={"help": "Optional input sequence length after tokenization."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=20,
        metadata={"help": "The percentage of the train set used as validation set"}
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the cached datasets"}
    )


@dataclass
class LoraArguments:
    """LoRA related arguments"""
    lora_r: int = field(default=4, metadata={"help": "Lora attention dimension"})
    lora_alpha: int = field(default=32, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.1, metadata={"help": "Lora dropout"})
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Lora target modules, separated by comma"}
    )
    lora_bias: str = field(default="none", metadata={"help": "Lora bias"})
    lora_task_type: str = field(default="CAUSAL_LM", metadata={"help": "Lora task type"})


def load_model_and_tokenizer(model_args: ModelArguments, training_args: TrainingArguments):
    """Load model and tokenizer"""
    logger.info(f"Loading model from {model_args.model_name_or_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True,
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    # Determine compute dtype
    compute_dtype = torch.float16
    if training_args.bf16:
        compute_dtype = torch.bfloat16
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=True,
        torch_dtype=compute_dtype,
        low_cpu_mem_usage=True,
    )
    
    # Move to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
        logger.info("Model moved to GPU")
    
    # Enable gradient checkpointing
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")
    
    return model, tokenizer


def setup_lora(model, lora_args: LoraArguments):
    """Set up and apply LoRA configuration"""
    logger.info("Setting up LoRA configuration")
    
    # LoRA configuration
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=lora_args.lora_target_modules.split(","),
        lora_dropout=lora_args.lora_dropout,
        bias=lora_args.lora_bias,
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Create LoRA model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Set model to trainable state
    model.enable_input_require_grads()
    
    return model


def format_alpaca(examples):
    """Convert Alpaca dataset to training format"""
    texts = []
    for i in range(len(examples["instruction"])):
        text = f"### Instruction:\n{examples['instruction'][i]}\n\n"
        if examples.get("input") and examples["input"][i]:
            text += f"### Input:\n{examples['input'][i]}\n\n"
        text += f"### Response:\n{examples['output'][i]}"
        texts.append(text)
    return {"text": texts}


def load_and_prepare_dataset(data_args: DataArguments, tokenizer, training_args: TrainingArguments):
    """Load and preprocess dataset"""
    logger.info("Loading and preparing dataset")
    
    if data_args.dataset_name is not None:
        # Use HuggingFace dataset
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=data_args.dataset_cache_dir,
        )
        
        # Process Alpaca format dataset
        if "alpaca" in data_args.dataset_name.lower():
            raw_datasets = raw_datasets.map(
                format_alpaca,
                batched=True,
                remove_columns=raw_datasets["train"].column_names,
                load_from_cache_file=not data_args.overwrite_cache,
            )
    else:
        # Use local files
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
        
        # Load JSONL files
        extension = data_args.train_file.split(".")[-1] if data_args.train_file else "json"
        if extension == "jsonl":
            extension = "json"
        raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=data_args.dataset_cache_dir)
    
    # Split training data if validation dataset is not provided
    if "validation" not in raw_datasets.keys():
        logger.info(f"Splitting training data: {100-data_args.validation_split_percentage}% train, {data_args.validation_split_percentage}% validation")
        split_dataset = raw_datasets["train"].train_test_split(
            test_size=data_args.validation_split_percentage / 100,
            seed=training_args.seed if hasattr(training_args, 'seed') else 42
        )
        raw_datasets = DatasetDict({
            "train": split_dataset["train"],
            "validation": split_dataset["test"]
        })
    
    # Check if we need to process the dataset format
    if "text" not in raw_datasets["train"].column_names:
        # Check if it's Alpaca format
        if "instruction" in raw_datasets["train"].column_names:
            logger.info("Processing Alpaca format dataset")
            for split in raw_datasets:
                raw_datasets[split] = raw_datasets[split].map(
                    format_alpaca,
                    batched=True,
                    remove_columns=raw_datasets[split].column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                )
        else:
            # Assume the first column contains the text
            first_col = raw_datasets["train"].column_names[0]
            logger.info(f"Using '{first_col}' column as text")
            for split in raw_datasets:
                raw_datasets[split] = raw_datasets[split].rename_column(first_col, "text")
    
    # Tokenization function
    def tokenize_function(examples):
        model_inputs = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=data_args.block_size,
        )
        
        model_inputs["labels"] = model_inputs["input_ids"].copy()
        
        return model_inputs
    
    # Tokenize datasets
    with training_args.main_process_first(desc="dataset map tokenization"):
        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )
    
    # Limit number of samples
    if data_args.max_train_samples is not None:
        tokenized_datasets["train"] = tokenized_datasets["train"].select(range(data_args.max_train_samples))
    
    if data_args.max_eval_samples is not None:
        tokenized_datasets["validation"] = tokenized_datasets["validation"].select(range(data_args.max_eval_samples))
    
    return tokenized_datasets


def main():
    # Parse arguments
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, LoraArguments, TrainingArguments))
    
    # Process arguments in SageMaker environment
    model_args, data_args, lora_args, training_args = parser.parse_args_into_dataclasses()
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(os.sys.stdout)],
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Enable gradient checkpointing
    training_args.gradient_checkpointing = True
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, training_args)
    
    # LoRA setup
    model = setup_lora(model, lora_args)
    
    logger.info("Model setup complete")
    
    # Prepare dataset
    tokenized_datasets = load_and_prepare_dataset(data_args, tokenizer, training_args)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Start training
    logger.info("Starting training")
    trainer.train()
    
    # Save model
    logger.info("Saving model")
    trainer.save_model()
    tokenizer.save_pretrained(training_args.output_dir)
    
    # Save only LoRA adapter
    model.save_pretrained(training_args.output_dir)
    logger.info(f"LoRA adapter saved to {training_args.output_dir}")
    
    # Copy inference.py to the model directory for SageMaker endpoint
    import shutil
    inference_src = os.path.join(os.path.dirname(__file__), "inference.py")
    inference_dst = os.path.join(training_args.output_dir, "code", "inference.py")
    
    # Create code directory if it doesn't exist
    os.makedirs(os.path.dirname(inference_dst), exist_ok=True)
    
    if os.path.exists(inference_src):
        shutil.copy2(inference_src, inference_dst)
        logger.info(f"Copied inference.py to {inference_dst}")
    else:
        logger.warning(f"inference.py not found at {inference_src}")
    
    # Also save a requirements.txt for the inference environment
    requirements_content = """transformers>=4.36.0
torch>=2.0.0
peft>=0.6.0
accelerate>=0.24.0
"""
    requirements_path = os.path.join(training_args.output_dir, "code", "requirements.txt")
    with open(requirements_path, 'w') as f:
        f.write(requirements_content)
    logger.info(f"Created requirements.txt at {requirements_path}")


if __name__ == "__main__":
    main()
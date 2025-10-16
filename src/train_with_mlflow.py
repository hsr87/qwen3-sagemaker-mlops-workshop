#!/usr/bin/env python3
"""
QWEN3-0.6B LoRA Fine-tuning on SageMaker with MLflow Tracking
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from time import gmtime, strftime

import boto3

import torch
print(f"Using PyTorch version: {torch.__version__}")

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback,
)
from datasets import load_dataset, DatasetDict
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType
)

# MLflow imports
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not available. Training will proceed without MLflow tracking.")

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


@dataclass
class MLflowArguments:
    """MLflow tracking related arguments"""
    mlflow_tracking_server_arn: Optional[str] = field(
        default=None,
        metadata={"help": "ARN of the SageMaker MLflow tracking server"}
    )
    mlflow_experiment_name: str = field(
        default="qwen3-lora-training",
        metadata={"help": "Name of the MLflow experiment"}
    )
    mlflow_run_name: Optional[str] = field(
        default=None,
        metadata={"help": "Name of the MLflow run. If None, auto-generated with timestamp"}
    )


class MLflowCallback(TrainerCallback):
    """Custom callback to log metrics to MLflow during training"""

    def __init__(self, mlflow_enabled: bool = False):
        self.mlflow_enabled = mlflow_enabled

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to MLflow when trainer logs"""
        if not self.mlflow_enabled or logs is None:
            return

        # Filter out non-numeric values
        metrics = {k: v for k, v in logs.items() if isinstance(v, (int, float))}

        if metrics:
            try:
                mlflow.log_metrics(metrics, step=state.global_step)
                logger.info(f"Logged metrics to MLflow at step {state.global_step}: {metrics}")
            except Exception as e:
                logger.warning(f"Failed to log metrics to MLflow: {e}")

    def on_train_end(self, args, state, control, **kwargs):
        """Log final training summary to MLflow"""
        if not self.mlflow_enabled:
            return

        try:
            # Log final metrics
            final_metrics = {
                "final_train_loss": state.log_history[-1].get("loss", 0) if state.log_history else 0,
                "total_steps": state.global_step,
                "total_epochs": state.epoch if state.epoch else 0,
            }
            mlflow.log_metrics(final_metrics)
            logger.info("Training completed. Final metrics logged to MLflow.")
        except Exception as e:
            logger.warning(f"Failed to log final metrics to MLflow: {e}")


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

    return model, lora_config


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

    return tokenized_datasets, raw_datasets

def main():
    # Parse arguments
    parser = transformers.HfArgumentParser((
        ModelArguments,
        DataArguments,
        LoraArguments,
        MLflowArguments,
        TrainingArguments
    ))

    # Process arguments in SageMaker environment
    model_args, data_args, lora_args, mlflow_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log environment information
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"Transformers version: {transformers.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

    # Enable gradient checkpointing
    training_args.gradient_checkpointing = True

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, training_args)

    # LoRA setup
    model, lora_config = setup_lora(model, lora_args)

    logger.info("Model setup complete")

    # Prepare dataset
    tokenized_datasets, raw_datasets = load_and_prepare_dataset(data_args, tokenizer, training_args)

    train_dataset_size = len(tokenized_datasets["train"])
    eval_dataset_size = len(tokenized_datasets["validation"])
    logger.info(f"Training dataset size: {train_dataset_size}")
    logger.info(f"Evaluation dataset size: {eval_dataset_size}")

    # Set up MLflow tracking
    # Get tracking server ARN from environment variable (preferred) or hyperparameters
    tracking_arn = mlflow_args.mlflow_tracking_server_arn

    if not MLFLOW_AVAILABLE:
        raise ImportError("MLflow is required but not available. Please install mlflow: pip install mlflow")

    # SageMaker MLflow: Get tracking server URL and set ARN as environment variable
    logger.info(f"MLflow tracking server ARN: {tracking_arn}")

    # Set tracking URI
    mlflow.set_tracking_uri(tracking_arn)
    logger.info(f"MLflow tracking URI configured successfully")

    # Set experiment
    mlflow.set_experiment(mlflow_args.mlflow_experiment_name)
    logger.info(f"MLflow experiment set to: {mlflow_args.mlflow_experiment_name}")

    # Generate run name if not provided
    if mlflow_args.mlflow_run_name is None:
        timestamp = strftime('%Y%m%d-%H%M%S', gmtime())
        mlflow_args.mlflow_run_name = f"qwen3-lora-training-{timestamp}"

    # Start MLflow run
    mlflow.start_run(run_name=mlflow_args.mlflow_run_name)
    logger.info(f"MLflow run started: {mlflow_args.mlflow_run_name}")

    # Log parameters
    params = {
        # Model parameters
        "model_name": model_args.model_name_or_path,
        "model_revision": model_args.model_revision,

        # LoRA parameters
        "lora_r": lora_args.lora_r,
        "lora_alpha": lora_args.lora_alpha,
        "lora_dropout": lora_args.lora_dropout,
        "lora_target_modules": lora_args.lora_target_modules,
        "lora_bias": lora_args.lora_bias,

        # Training parameters
        "num_train_epochs": training_args.num_train_epochs,
        "per_device_train_batch_size": training_args.per_device_train_batch_size,
        "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "learning_rate": training_args.learning_rate,
        "weight_decay": training_args.weight_decay,
        "warmup_ratio": training_args.warmup_ratio,
        "lr_scheduler_type": training_args.lr_scheduler_type,
        "bf16": training_args.bf16,
        "gradient_checkpointing": training_args.gradient_checkpointing,

        # Data parameters
        "block_size": data_args.block_size,
        "validation_split_percentage": data_args.validation_split_percentage,
        "train_dataset_size": train_dataset_size,
        "eval_dataset_size": eval_dataset_size,
    }

    mlflow.log_params(params)
    logger.info(f"Logged {len(params)} parameters to MLflow")

    # Log LoRA configuration as artifact
    lora_config_dict = {
        "r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
        "target_modules": list(lora_config.target_modules),
        "bias": lora_config.bias,
        "task_type": str(lora_config.task_type),
    }

    # Save LoRA config to temp file and log as artifact
    lora_config_path = "/tmp/lora_config.json"
    with open(lora_config_path, "w") as f:
        json.dump(lora_config_dict, f, indent=2)
    mlflow.log_artifact(lora_config_path, artifact_path="config")
    logger.info("Logged LoRA configuration as artifact")

    # Setup trainer with MLflow callback
    callbacks = []
    callbacks.append(MLflowCallback(mlflow_enabled=True))


    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )

    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
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

    # Log model artifacts to MLflow
    try:
        # Log the final model to MLflow
        logger.info("Logging model artifacts to MLflow...")
        mlflow.log_artifacts(training_args.output_dir, artifact_path="model")
        logger.info("Model artifacts logged to MLflow")
    except Exception as e:
        logger.warning(f"Failed to log model artifacts to MLflow: {e}")

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

    # End MLflow run
    try:
        mlflow.end_run()
        logger.info("MLflow run ended successfully")
    except Exception as e:
        logger.warning(f"Failed to end MLflow run: {e}")

    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()

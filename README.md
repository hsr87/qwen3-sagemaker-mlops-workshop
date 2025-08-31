# Qwen3 SageMaker MLOps Workshop

A comprehensive MLOps pipeline workshop using Amazon SageMaker to fine-tune and deploy the Qwen3-0.6B model.

## Overview

This workshop demonstrates how to build a complete MLOps pipeline using the Qwen3-0.6B model, from fine-tuning on Korean text data to deployment on Amazon SageMaker.

## Workshop Components

### 1. Model Training
- **Notebook**: `1_training.ipynb`
- **Script**: `src/train.py`
- Fine-tune Qwen3-0.6B model using LoRA approach
- Uses Korean food description text data

### 2. Model Evaluation
- **Notebook**: `2_evaluation.ipynb`
- **Script**: `src/evaluation/evaluate.py`
- Model performance evaluation using SageMaker Processing Job

### 3. Model Registry
- **Notebook**: `3_model_registry.ipynb`
- Register models that meet performance criteria to Model Registry
- Model version management and approval process

### 4. Endpoint Deployment
- **Notebook**: `4_endpoint_deployment.ipynb`
- **Script**: `src/inference.py`
- Deploy registered models to SageMaker Endpoints

### 5. SageMaker Pipeline
- **Notebook**: `5_sagemaker_pipeline.ipynb`
- Automated MLOps pipeline orchestrating the entire process

## Dataset

### Training Data
- **File**: `samples/train.jsonl`
- Korean food description texts
- JSONL format, each line contains `{"text": "..."}`

### Test Data
- **File**: `samples/test.jsonl`
- Test data for model evaluation
- Same format as training data

## Project Structure

```
qwen3-sagemaker-mlops-workshop/
├── README.md
├── CLAUDE.md                          # Project settings and guidelines
├── 1_training.ipynb                   # Model training notebook
├── 2_evaluation.ipynb                 # Model evaluation notebook
├── 3_model_registry.ipynb             # Model registry notebook
├── 4_endpoint_deployment.ipynb        # Endpoint deployment notebook
├── 5_sagemaker_pipeline.ipynb         # SageMaker pipeline notebook
├── samples/
│   ├── train.jsonl                    # Training data
│   └── test.jsonl                     # Test data
├── src/
│   ├── train.py                       # Training script
│   ├── inference.py                   # Inference script
│   ├── requirements.txt               # Python dependencies
│   └── evaluation/
│       ├── evaluate.py                # Evaluation script
│       └── requirements.txt           # Evaluation dependencies
├── endpoint_info.json                 # Endpoint information
└── latest_model_arn.txt              # Latest model ARN
```

## Prerequisites

### AWS Environment
- Amazon SageMaker Notebook Instance or SageMaker Studio
- Proper IAM permissions (SageMaker execution role)
- Access to GPU instances (ml.g5.2xlarge recommended)

### Python Packages
- transformers
- torch
- peft
- datasets
- evaluate
- boto3
- sagemaker

## Getting Started

1. Open the project in your SageMaker Jupyter Notebook environment
2. Run the notebooks in order:
   - `1_training.ipynb`: Model training
   - `2_evaluation.ipynb`: Model evaluation
   - `3_model_registry.ipynb`: Model registry registration
   - `4_endpoint_deployment.ipynb`: Endpoint deployment
   - `5_sagemaker_pipeline.ipynb`: Pipeline configuration

## Key Features

- **LoRA Fine-tuning**: Memory-efficient Low-Rank Adaptation
- **Korean Language Data**: Domain-specific training on Korean food texts
- **Complete MLOps**: End-to-end pipeline automation from training to deployment
- **Model Governance**: Model version management through Model Registry
- **Real-time Inference**: Real-time serving via SageMaker Endpoints

## Resource Costs

- **Training**: ml.g5.2xlarge (GPU instance)
- **Processing**: ml.m5.xlarge (for evaluation)
- **Endpoint**: ml.g5.2xlarge (for inference)

Remember to delete endpoints after the workshop to save costs.

## Troubleshooting

Each notebook contains detailed execution results and logs. If you encounter issues, check:

1. IAM permissions configuration
2. Instance type availability
3. S3 bucket access permissions
4. Service quotas and limits

## References
- [Amazon SageMaker Developer Guide](https://docs.aws.amazon.com/sagemaker/)
- [Qwen3 Model Information](https://huggingface.co/Qwen/Qwen3-0.6B)
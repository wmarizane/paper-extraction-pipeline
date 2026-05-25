#!/bin/bash
#SBATCH --job-name=judge-eval
#SBATCH --output=logs/eval_%j.log
#SBATCH --error=logs/eval_%j.err
#SBATCH --time=04:00:00
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:rtx_6000:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

# Load environment
source /project/wkmrzane/miniconda3/bin/activate research-assistant

# Set offline mode and cache dirs for air-gapped environment
export HF_HUB_OFFLINE=1
export TIKTOKEN_CACHE_DIR=/project/wkmrzane/.cache/tiktoken
export HF_HOME=/project/wkmrzane/.cache/huggingface

echo "==============================================="
echo "Starting Position Bias Evaluation for Judge Models"
echo "Date: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "==============================================="

export PYTHONPATH=$PYTHONPATH:$(pwd)
python -m evaluation.judge_evaluator --models deepseek-r1-32b qwen3.5-27b mistral-small-24b

echo "==============================================="
echo "Evaluation completed at $(date)"
echo "==============================================="

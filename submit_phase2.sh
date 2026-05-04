#!/bin/bash
# Submit Phase 2 extractions across multiple models in parallel

echo "🚀 Submitting Phase 2 Multi-Model Evaluation..."

# Ensure we are in the right directory
cd /project/wkmrzane/research-assistant/paper-extraction-pipeline

models=(
    "qwen3.5-27b:1"       # 1 GPU
    "deepseek-r1-32b:2"   # 2 GPUs
    "llama3.3-70b:4"      # 4 GPUs
)

for entry in "${models[@]}"; do
    IFS=':' read -r model gpus <<< "$entry"
    echo "Submitting job for: $model with $gpus GPUs"
    sbatch --export=ALL,MODEL=$model --gres=gpu:rtx_6000:$gpus run_extraction.slurm
done

echo "✅ All models submitted. Run 'squeue -u $USER' to monitor progress."

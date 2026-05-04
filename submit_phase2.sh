#!/bin/bash
# Submit Phase 2 extractions across multiple models in parallel

echo "🚀 Submitting Phase 2 Multi-Model Evaluation..."

# Ensure we are in the right directory
cd /project/wkmrzane/research-assistant/paper-extraction-pipeline

models=(
    "qwen3.5-27b"
    "deepseek-r1-32b"
    "llama3.3-70b"
)

for model in "${models[@]}"; do
    echo "Submitting job for: $model"
    sbatch --export=ALL,MODEL=$model run_extraction.slurm
done

echo "✅ All models submitted. Run 'squeue -u $USER' to monitor progress."

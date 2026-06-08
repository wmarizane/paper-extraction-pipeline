#!/bin/bash
# Submit all models and subfolders to SLURM

set -e

MODELS=("qwen3.5-27b" "mistral-small-24b")
SUBFOLDERS=()

# Find all subfolders in Inputs
for SUBDIR in Inputs/*/; do
    [ -d "$SUBDIR" ] || continue
    SUBFOLDERS+=("$(basename "$SUBDIR")")
done

if [ ${#SUBFOLDERS[@]} -eq 0 ]; then
    echo "No subfolders found in Inputs/."
    exit 1
fi

echo "Found ${#SUBFOLDERS[@]} subfolders: ${SUBFOLDERS[*]}"
echo "Models: ${MODELS[*]}"
echo ""

JOB_IDS=()

for MODEL in "${MODELS[@]}"; do
    for SUBFOLDER in "${SUBFOLDERS[@]}"; do
        echo "Submitting extraction job for $MODEL on $SUBFOLDER..."
        
        # Mistral needs 2 GPUs (tensor_parallel_size=2)
        if [ "$MODEL" == "mistral-small-24b" ]; then
            GPU_REQ="gpu:rtx_6000:2"
        else
            GPU_REQ="gpu:rtx_6000:1"
        fi
        
        # Extract job ID from parsable output
        JOB_ID=$(sbatch --parsable --gres=$GPU_REQ --export=ALL,MODEL=$MODEL,SUBFOLDER=$SUBFOLDER run_extraction.slurm)
        echo "  -> Job ID: $JOB_ID (GPUs: $GPU_REQ)"
        JOB_IDS+=("$JOB_ID")
    done
done

# Join job IDs with colons for the dependency string
DEPENDENCY_STR=$(IFS=:; echo "${JOB_IDS[*]}")

echo ""
echo "Submitting Consensus Judge with dependencies on all extraction jobs..."
sbatch --dependency=afterok:$DEPENDENCY_STR submit_phase3.sh

echo "All jobs submitted!"

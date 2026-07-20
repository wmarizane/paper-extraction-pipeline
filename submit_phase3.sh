#!/bin/bash
#SBATCH --job-name=phase3-consensus
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:rtx_6000:2
#SBATCH --time=08:00:00
#SBATCH --output=logs/output_%x-%j.log
#SBATCH --error=logs/error_%x-%j.log

set -e

mkdir -p logs results/consensus

source /project/wkmrzane/miniconda3/etc/profile.d/conda.sh
conda activate research-assistant

PROJECT_DIR="/project/wkmrzane/research-assistant/paper-extraction-pipeline"
cd "$PROJECT_DIR" || exit 1

export PYTHONPATH=.
# Unbuffered stdout so per-paper progress is visible live in the .log (a killed
# job otherwise loses its entire buffered progress trail — see job 151555).
export PYTHONUNBUFFERED=1

# Library paths
export LD_LIBRARY_PATH=/project/wkmrzane/miniconda3/envs/research-assistant/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/project/wkmrzane/miniconda3/envs/research-assistant/lib/stubs:$LIBRARY_PATH
export LD_PRELOAD=/project/wkmrzane/miniconda3/envs/research-assistant/lib/libstdc++.so.6

if [ -f ".env" ]; then
    export HF_HUB_OFFLINE=1
    set -a
    source <(grep -v '^#' .env | sed 's/#.*//' | sed 's/[[:space:]]*$//')
    set +a
fi

echo "================================"
echo "Starting Phase 3: Consensus Judge"
echo "================================"

python run_consensus.py

echo "================================"
echo "Exporting CSV Summaries for all subfolders"
echo "================================"
for SUBDIR in results/consensus/*/; do
    [ -d "$SUBDIR" ] || continue
    SUBFOLDER=$(basename "$SUBDIR")
    echo "Exporting CSVs for subfolder: $SUBFOLDER"
    
    if [ -d "results/qwen3.5-27b/$SUBFOLDER" ]; then
        python pipeline/csv_exporter.py "results/qwen3.5-27b/$SUBFOLDER" "results/${SUBFOLDER}_qwen_summary.csv"
    fi
    if [ -d "results/mistral-small-24b/$SUBFOLDER" ]; then
        python pipeline/csv_exporter.py "results/mistral-small-24b/$SUBFOLDER" "results/${SUBFOLDER}_mistral_summary.csv"
    fi
    python pipeline/csv_exporter.py "$SUBDIR" "results/${SUBFOLDER}_consensus_summary.csv"
done

echo "================================"
echo "Job Complete"
echo "================================"

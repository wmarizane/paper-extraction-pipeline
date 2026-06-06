#!/bin/bash
#SBATCH --job-name=phase3-consensus
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:rtx_6000:2
#SBATCH --time=04:00:00
#SBATCH --output=logs/output_%x-%j.log
#SBATCH --error=logs/error_%x-%j.log

set -e

mkdir -p logs results/consensus

source /project/wkmrzane/miniconda3/etc/profile.d/conda.sh
conda activate research-assistant

PROJECT_DIR="/project/wkmrzane/research-assistant/paper-extraction-pipeline"
cd "$PROJECT_DIR" || exit 1

export PYTHONPATH=.

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
echo "Job Complete"
echo "================================"

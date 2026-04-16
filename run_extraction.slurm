#!/bin/bash
#SBATCH --job-name=paper-extract
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/output_%x-%j.log
#SBATCH --error=logs/error_%x-%j.log

# ============================================================================
# Paper Extraction Pipeline - Phase 1 Refactor (vLLM + PyMuPDF)
# ============================================================================

set -e  # Exit on error

mkdir -p logs results

# Conda Environment Setup
source /project/wkmrzane/miniconda3/etc/profile.d/conda.sh
conda activate research-assistant
NLP_PYTHON=$(which python)
echo "Python: $NLP_PYTHON"
echo "Python version: $($NLP_PYTHON --version)"

echo "Installing pymupdf4llm..."
$NLP_PYTHON -m pip install --quiet --no-cache-dir pymupdf4llm

PROJECT_DIR="/project/wkmrzane/research-assistant/paper-extraction-pipeline"
cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd to $PROJECT_DIR"; exit 1; }
echo "Working in: $(pwd)"


if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    set -a
    source <(grep -v '^#' .env | sed 's/#.*//' | sed 's/[[:space:]]*$//')
    set +a
fi

echo "================================"
echo "Starting Extraction Pipeline"
echo "================================"

export PYTHONPATH=.

# Using run_local.py to test the pipeline on polymerPaper1.pdf
# For batch processing across the inputs folder, you can use a loop later.
$NLP_PYTHON run_local.py Inputs/polymerPaper1.pdf

PIPELINE_EXIT_CODE=$?

echo "================================"
echo "Job Summary"
echo "================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "End Time: $(date)"
echo "Exit Code: ${PIPELINE_EXIT_CODE}"
echo "================================"

exit ${PIPELINE_EXIT_CODE}

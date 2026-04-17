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

export LD_LIBRARY_PATH=/project/wkmrzane/miniconda3/envs/research-assistant/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/project/wkmrzane/miniconda3/envs/research-assistant/lib/stubs:$LIBRARY_PATH
export LD_PRELOAD=/project/wkmrzane/miniconda3/envs/research-assistant/lib/libstdc++.so.6

# Delete stale FlashInfer GDN kernel cache so it recompiles with correct libstdc++
echo "Clearing stale FlashInfer kernel cache..."
rm -rf /project/wkmrzane/.cache/flashinfer/
echo "Cache cleared."

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

# Process all papers in Inputs/
PIPELINE_EXIT_CODE=0
for PDF in Inputs/polymerPaper1.pdf Inputs/polymerPaper2.pdf Inputs/polymerPaper3.pdf; do
    echo ""
    echo ">>> Processing: $PDF"
    $NLP_PYTHON run_local.py "$PDF"
    EXIT=$?
    if [ $EXIT -ne 0 ]; then
        echo "WARNING: $PDF failed with exit code $EXIT"
        PIPELINE_EXIT_CODE=$EXIT
    fi
done

echo "================================"
echo "Job Summary"
echo "================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "End Time: $(date)"
echo "Exit Code: ${PIPELINE_EXIT_CODE}"
echo "================================"

exit ${PIPELINE_EXIT_CODE}

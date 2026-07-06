#!/bin/bash
#SBATCH --job-name=qwen-rerun-25
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:rtx_6000:1
#SBATCH --time=12:00:00
#SBATCH --output=logs/output_%x-%j.log
#SBATCH --error=logs/error_%x-%j.log

set -e  # Exit on error

mkdir -p logs results

# Conda Environment Setup
source /project/wkmrzane/miniconda3/etc/profile.d/conda.sh
conda activate research-assistant
NLP_PYTHON=$(which python)
echo "Python: $NLP_PYTHON"

PROJECT_DIR="/project/wkmrzane/research-assistant/paper-extraction-pipeline"
cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd to $PROJECT_DIR"; exit 1; }
echo "Working in: $(pwd)"

export LD_LIBRARY_PATH=/project/wkmrzane/miniconda3/envs/research-assistant/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/project/wkmrzane/miniconda3/envs/research-assistant/lib/stubs:$LIBRARY_PATH
export LD_PRELOAD=/project/wkmrzane/miniconda3/envs/research-assistant/lib/libstdc++.so.6

if [ -f ".env" ]; then
    echo "Loading environment variables from .env..."
    set -a
    source <(grep -v '^#' .env | sed 's/#.*//' | sed 's/[[:space:]]*$//')
    set +a
fi

export PYTHONPATH=.
export HF_HUB_OFFLINE=1
export TIKTOKEN_CACHE_DIR=/project/wkmrzane/.cache/tiktoken
export FLASHINFER_WORKSPACE_BASE="/project/wkmrzane/.cache/flashinfer/${SLURM_JOB_ID:-standalone}"
mkdir -p "$FLASHINFER_WORKSPACE_BASE"

MODEL="qwen3.5-27b"
echo "Model override: $MODEL"
MODEL_FLAG="--model $MODEL"

PIPELINE_EXIT_CODE=0

declare -a FAILED_PDFS=(
    "Inputs/PEG/[107] Krüger1996.pdf"
    "Inputs/PEG/[111] Gorshkov1990.pdf"
    "Inputs/PEG/[112] Pasch1995.pdf"
    "Inputs/PEG/[113] Pasch.pdf"
    "Inputs/PEG/[115] Falkenhagen2000.pdf"
    "Inputs/PEG/[117] Murgasova1998.pdf"
    "Inputs/PEG/[118] Trathnigg1994.pdf"
    "Inputs/PEG/[120] Trathnigg2001.pdf"
    "Inputs/PEG/[121] Lochmuller1996.pdf"
    "Inputs/PLA/[148] Wachsen.pdf"
    "Inputs/PLA/[210] Biela2003.pdf"
    "Inputs/PLA/[403] Radke2005.pdf"
    "Inputs/PLA/[94] Krüger1996.pdf"
    "Inputs/PPO/[125] Gorbunov1997.pdf"
    "Inputs/PPO/[127] Esser2001.pdf"
    "Inputs/PPO/[128] Gorshkov1986.pdf"
    "Inputs/PPO/[209] Bhati2016.pdf"
    "Inputs/PPO/[232] Malik2009.pdf"
    "Inputs/PPO/[251] Trathnigg2005.pdf"
    "Inputs/PPO/[252] Trathnigg2007.pdf"
    "Inputs/PPO/[256] Zdovc2022.pdf"
    "Inputs/PPO/[294] Malik2008.pdf"
    "Inputs/PPO/[295] Malik2009.pdf"
    "Inputs/PPO/[321] Malik2009.pdf"
    "Inputs/PPO/[323] Malik2016.pdf"
)

for PDF in "${FAILED_PDFS[@]}"; do
    if [ ! -f "$PDF" ]; then
        echo "WARNING: File not found: $PDF"
        continue
    fi
    # Extract subfolder from PDF path (e.g., Inputs/PEG/... -> PEG)
    SUBFOLDER=$(basename $(dirname "$PDF"))
    echo ""
    echo ">>> Processing: $PDF (Subfolder: $SUBFOLDER)"
    set +e
    $NLP_PYTHON run_local.py "$PDF" $MODEL_FLAG --subfolder "$SUBFOLDER"
    EXIT=$?
    set -e
    if [ $EXIT -ne 0 ]; then
        echo "WARNING: $PDF failed with exit code $EXIT"
        PIPELINE_EXIT_CODE=$EXIT
    fi
done

echo "================================"
echo "Job Summary"
echo "================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Model: $MODEL"
echo "End Time: $(date)"
echo "Exit Code: ${PIPELINE_EXIT_CODE}"
echo "================================"

exit ${PIPELINE_EXIT_CODE}

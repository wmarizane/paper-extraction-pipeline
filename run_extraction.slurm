#!/bin/bash

#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=bigTiger
#SBATCH --gres=gpu:1
#SBATCH --time=0-06:00:00
#SBATCH --job-name=paper-extract
#SBATCH --output=logs/output_extraction-%j.log
#SBATCH --error=logs/error_extraction-%j.log

# ==============================================================================
# Paper Extraction Pipeline - SLURM Script for iTiger HPC
# ==============================================================================
# PDF → GROBID → Chunking → LLM Extraction (Qwen 3.5) → JSON Output
# ==============================================================================

# Record start time
START_TIME=$(date +%s)
START_DATETIME=$(date)

# Print job info
echo "=========================================="
echo "Paper Extraction Pipeline"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $START_DATETIME"
echo "Working Directory: $(pwd)"
echo "=========================================="

# Make sure the logs directory exists
mkdir -p logs
mkdir -p results

# ------------------------------------------------------------------------------
# Conda Environment Setup
# ------------------------------------------------------------------------------
ENV_NAME="nlp"

source ~/.bashrc
source "$(conda info --base)/etc/profile.d/conda.sh"

echo ""
echo "Activating environment '$ENV_NAME'"
conda activate $ENV_NAME
echo "Using conda env: $CONDA_DEFAULT_ENV"

# Use explicit nlp Python to guarantee correct environment
NLP_PYTHON=/project/wkmrzane/miniconda3/envs/nlp/bin/python
echo "Python: $NLP_PYTHON"
echo "Python version: $($NLP_PYTHON --version)"

# Ensure required packages are installed (if not already in conda env)
echo ""
echo "Verifying required packages..."
$NLP_PYTHON -m pip install --quiet --no-cache-dir \
    pydantic \
    pydantic-settings \
    python-dotenv \
    tiktoken \
    langchain-ollama \
    requests \
    lxml

echo "Package verification complete"

# ------------------------------------------------------------------------------
# Navigate to project directory
# ------------------------------------------------------------------------------
PROJECT_DIR="/project/wkmrzane/reserach-assistant/paper-extraction-pipeline"
cd "$PROJECT_DIR" || { echo "ERROR: Cannot cd to $PROJECT_DIR"; exit 1; }
echo "Working in: $(pwd)"

# ------------------------------------------------------------------------------
# Load environment variables from .env
# ------------------------------------------------------------------------------
if [ -f ".env" ]; then
    echo ""
    echo "Loading environment variables from .env..."
    set -a
    source <(grep -v '^#' .env | sed 's/#.*//' | sed 's/[[:space:]]*$//')
    set +a
else
    echo "WARNING: .env file not found! Using .env.example..."
    if [ -f ".env.example" ]; then
        cp .env.example .env
        set -a
        source <(grep -v '^#' .env | sed 's/#.*//' | sed 's/[[:space:]]*$//')
        set +a
    else
        echo "ERROR: Neither .env nor .env.example found!"
        exit 1
    fi
fi

echo ""
echo "Configuration:"
echo "  LLM_MODEL: $LLM_MODEL"
echo "  CHUNK_STRATEGY: $CHUNK_STRATEGY"
echo "  CHUNK_SIZE: $CHUNK_SIZE"
echo "  GROBID_URL: $GROBID_URL"
echo ""

# Display GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# ------------------------------------------------------------------------------
# Ollama Setup (GPU-accelerated)
# ------------------------------------------------------------------------------
export OLLAMA_HOME=/project/wkmrzane/.ollama
export OLLAMA_MODELS=/project/wkmrzane/.ollama/models
export PATH=/project/wkmrzane/miniconda3/bin:$PATH

# Use a unique port per job to avoid conflicts with other Ollama jobs on same node
OLLAMA_PORT=$((11434 + (SLURM_JOB_ID % 1000)))
export OLLAMA_HOST="127.0.0.1:${OLLAMA_PORT}"
export OLLAMA_URL="http://127.0.0.1:${OLLAMA_PORT}"

# Update .env with the correct Ollama port for this job
sed -i "s|OLLAMA_URL=.*|OLLAMA_URL=http://localhost:${OLLAMA_PORT}|" .env

echo "Ollama Configuration:"
echo "  OLLAMA_PORT: $OLLAMA_PORT"
echo "  OLLAMA_HOST: $OLLAMA_HOST"
echo "  OLLAMA_MODELS: $OLLAMA_MODELS"
echo ""

echo "Starting Ollama server..."
ollama serve > logs/ollama_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID"

# Wait for Ollama to be ready
echo "Waiting for Ollama to start..."
for i in {1..30}; do
    if curl -s http://127.0.0.1:${OLLAMA_PORT}/api/version > /dev/null 2>&1; then
        echo "✓ Ollama server is running"
        curl -s http://127.0.0.1:${OLLAMA_PORT}/api/version
        echo ""
        break
    fi
    if [ $i -eq 30 ]; then
        echo "ERROR: Ollama server failed to start after 30 seconds"
        cat logs/ollama_${SLURM_JOB_ID}.log
        exit 1
    fi
    sleep 1
done

# Pull the model if not already downloaded
echo ""
echo "Ensuring model '${LLM_MODEL}' is available..."
ollama pull ${LLM_MODEL}

echo ""
echo "Available models:"
ollama list
echo ""

# ------------------------------------------------------------------------------
# GROBID Setup
# ------------------------------------------------------------------------------
# Check if GROBID is accessible
echo "Checking GROBID server at ${GROBID_URL}..."
if curl -s -f "${GROBID_URL}/api/version" > /dev/null 2>&1; then
    echo "✓ GROBID server is accessible"
    curl -s "${GROBID_URL}/api/version"
    echo ""
else
    echo "WARNING: GROBID server at ${GROBID_URL} is not accessible"
    echo "The pipeline will fail if GROBID is required."
    echo "Make sure GROBID is running or update GROBID_URL in .env"
    echo ""
    # Don't exit - let the pipeline handle the error
fi

# ------------------------------------------------------------------------------
# Run Parameters
# ------------------------------------------------------------------------------
# Default to processing all PDFs in Inputs/ directory
INPUT_PDFS="${INPUT_PDFS:-Inputs/*.pdf}"
MAX_PDFS="${MAX_PDFS:-0}"  # 0 = no limit

echo ""
echo "=========================================="
echo "Pipeline Configuration:"
echo "  Input PDFs: $INPUT_PDFS"
echo "  Model: $LLM_MODEL"
echo "  Max PDFs: $MAX_PDFS (0 = unlimited)"
echo "  Output Dir: results/"
echo "=========================================="
echo ""

# Verify input files exist
if ! ls $INPUT_PDFS 1> /dev/null 2>&1; then
    echo "ERROR: No input PDF files found matching: $INPUT_PDFS"
    exit 1
fi

PDF_COUNT=$(ls $INPUT_PDFS 2>/dev/null | wc -l)
echo "Found $PDF_COUNT PDF file(s) to process"
echo ""

# ------------------------------------------------------------------------------
# Run Extraction Pipeline
# ------------------------------------------------------------------------------
echo "Starting paper extraction pipeline..."
echo "================================================================"
echo ""

# Set PYTHONPATH to include current directory
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"

# Run the pipeline
$NLP_PYTHON run_local.py $INPUT_PDFS

# Capture exit code
EXIT_CODE=$?

# ------------------------------------------------------------------------------
# Post-processing & Cleanup
# ------------------------------------------------------------------------------
END_TIME=$(date +%s)
END_DATETIME=$(date)
TOTAL_SECONDS=$((END_TIME - START_TIME))
HOURS=$((TOTAL_SECONDS / 3600))
MINUTES=$(((TOTAL_SECONDS % 3600) / 60))
SECONDS=$((TOTAL_SECONDS % 60))

echo ""
echo "=========================================="
echo "Pipeline completed with exit code: $EXIT_CODE"
echo "Start Time:  $START_DATETIME"
echo "End Time:    $END_DATETIME"
echo "Total Time:  ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=========================================="

# Print results summary
echo ""
echo "Results Summary:"
RESULT_COUNT=$(ls results/*_latest.json 2>/dev/null | wc -l)
echo "  Output files: $RESULT_COUNT"
if [ $RESULT_COUNT -gt 0 ]; then
    echo "  Latest results:"
    ls -lh results/*_latest.json
fi
echo ""

# Cleanup: stop Ollama server
echo "Stopping Ollama server (PID: $OLLAMA_PID)..."
kill $OLLAMA_PID 2>/dev/null

# Give Ollama time to clean up
sleep 2

echo "Cleanup complete"
echo ""

# Exit with the pipeline's exit code
exit $EXIT_CODE

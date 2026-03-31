#!/bin/bash
#SBATCH --job-name=paper-extract
#SBATCH --partition=bigTiger
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=logs/output_%x-%j.log
#SBATCH --error=logs/error_%x-%j.log

# ============================================================================
# Paper Extraction Pipeline - vLLM Version
# ============================================================================
# 
# This script runs the PDF extraction pipeline on the cluster using vLLM
# for LLM-based extraction with GPU acceleration.
#
# Key changes from Ollama version:
#   - No Ollama server management (vLLM loads models directly in Python)
#   - Simplified script (no port conflicts, no background processes)
#   - Faster inference with better GPU utilization
#
# Usage:
#   sbatch run_extraction.slurm
#
# ============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# ============================================================================
# Environment Setup
# ============================================================================

echo "================================"
echo "SLURM Job Information"
echo "================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Node: ${SLURM_NODELIST}"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Memory: ${SLURM_MEM_PER_NODE}MB"
echo "Start Time: $(date)"
echo "================================"
echo ""

# ============================================================================
# Conda Environment
# ============================================================================

echo "Activating conda environment..."
source /project/wkmrzane/miniconda3/etc/profile.d/conda.sh
conda activate research-assistant

# Verify Python and vLLM
echo ""
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "vLLM installed: $(python -c 'import vllm; print(vllm.__version__)')"
echo ""

# ============================================================================
# GPU Check
# ============================================================================

echo "================================"
echo "GPU Information"
echo "================================"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Test CUDA availability
python << 'PYTEST'
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("ERROR: CUDA not available!")
    exit(1)
PYTEST

echo ""

# ============================================================================
# GROBID Server Setup
# ============================================================================

echo "================================"
echo "Starting GROBID Server"
echo "================================"

# Calculate unique port based on job ID to avoid conflicts
GROBID_PORT=$((8070 + (SLURM_JOB_ID % 1000)))
echo "GROBID port: ${GROBID_PORT}"

# Path to Singularity image
GROBID_IMAGE="/project/wkmrzane/.singularity/grobid_0.8.0.sif"

if [ ! -f "${GROBID_IMAGE}" ]; then
    echo "ERROR: GROBID Singularity image not found at ${GROBID_IMAGE}"
    echo "Please build it first with:"
    echo "  singularity pull ${GROBID_IMAGE} docker://lfoppiano/grobid:0.8.0"
    exit 1
fi

# Create GROBID config with dynamic port
cat > grobid_config.yaml << GROBID_EOF
grobid:
  server:
    port: ${GROBID_PORT}
GROBID_EOF

# Start GROBID in background
echo "Starting GROBID with Singularity..."
singularity exec \
    --env GROBID_HOME=/opt/grobid/grobid-home \
    --bind ${PWD}/grobid_config.yaml:/config/grobid.yaml \
    "${GROBID_IMAGE}" \
    /opt/grobid/grobid-service/bin/grobid-service server /config/grobid.yaml \
    > logs/grobid_${SLURM_JOB_ID}.log 2>&1 &

GROBID_PID=$!
echo "GROBID PID: ${GROBID_PID}"

# Wait for GROBID to be ready
echo "Waiting for GROBID to start..."
GROBID_URL="http://localhost:${GROBID_PORT}"
MAX_WAIT=120
WAITED=0

while ! curl -s "${GROBID_URL}/api/version" > /dev/null; do
    if [ $WAITED -ge $MAX_WAIT ]; then
        echo "ERROR: GROBID failed to start after ${MAX_WAIT}s"
        echo "GROBID logs:"
        tail -50 logs/grobid_${SLURM_JOB_ID}.log
        kill ${GROBID_PID} 2>/dev/null || true
        exit 1
    fi
    sleep 2
    WAITED=$((WAITED + 2))
    echo -n "."
done

echo ""
echo "✅ GROBID is ready!"
GROBID_VERSION=$(curl -s "${GROBID_URL}/api/version")
echo "GROBID version: ${GROBID_VERSION}"
echo ""

# Update .env with GROBID URL
if [ -f .env ]; then
    sed -i "s|^GROBID_URL=.*|GROBID_URL=${GROBID_URL}|" .env
else
    echo "GROBID_URL=${GROBID_URL}" > .env
fi

# ============================================================================
# Run Extraction Pipeline
# ============================================================================

echo "================================"
echo "Starting Extraction Pipeline"
echo "================================"
echo "Model: ${LLM_MODEL:-Qwen/Qwen2.5-27B-Instruct}"
echo "Input directory: ${INPUT_DIR:-./inputs}"
echo "Output directory: ${OUTPUT_DIR:-./results}"
echo ""

# Run pipeline with unbuffered output
python -u run_local.py

PIPELINE_EXIT_CODE=$?

echo ""
echo "================================"
echo "Pipeline Execution Complete"
echo "================================"
echo "Exit code: ${PIPELINE_EXIT_CODE}"
echo ""

# ============================================================================
# Cleanup
# ============================================================================

echo "Cleaning up..."

# Stop GROBID
if [ ! -z "${GROBID_PID}" ]; then
    echo "Stopping GROBID (PID: ${GROBID_PID})..."
    kill ${GROBID_PID} 2>/dev/null || true
    wait ${GROBID_PID} 2>/dev/null || true
fi

# Remove temporary config
rm -f grobid_config.yaml

echo "✅ Cleanup complete"
echo ""

# ============================================================================
# Job Summary
# ============================================================================

echo "================================"
echo "Job Summary"
echo "================================"
echo "Job ID: ${SLURM_JOB_ID}"
echo "End Time: $(date)"
echo "Duration: $SECONDS seconds"
echo "Exit Code: ${PIPELINE_EXIT_CODE}"
echo "================================"

exit ${PIPELINE_EXIT_CODE}

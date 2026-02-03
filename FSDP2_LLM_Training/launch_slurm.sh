#!/bin/bash
# SLURM multi-node training launcher
# Usage: sbatch launch_slurm.sh

#SBATCH --job-name=llm-training
#SBATCH --nodes=4                    # Number of nodes
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8            # GPUs per node
#SBATCH --cpus-per-task=96
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --exclusive

# ============================================================================
# Configuration
# ============================================================================

# Model configuration - Choose one approach:
# Option 1: Use Hugging Face model
USE_HF_MODEL=true
HF_MODEL="meta-llama/Llama-2-70b-hf"

# Option 2: Train from scratch (set USE_HF_MODEL=false)
MODEL_SIZE="70B"

# Training configuration
BATCH_SIZE=2
GRAD_ACCUM_STEPS=8
MAX_STEPS=100000
SEQ_LEN=2048

# Optimizer
MAX_LR=1.5e-4
MIN_LR=1.5e-5
WARMUP_STEPS=2000

# Checkpointing
CHECKPOINT_DIR="/shared/checkpoints/llm_$(date +%Y%m%d_%H%M%S)"

# ============================================================================
# Environment Setup
# ============================================================================

mkdir -p logs

# Load modules (adjust for your cluster)
# module load cuda/12.1
# module load nccl
# module load python/3.10

# Activate environment
# source activate llm-training

# NCCL configuration
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0
export NCCL_SOCKET_IFNAME=eth0

# PyTorch configuration
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# ============================================================================
# Distributed Setup
# ============================================================================

MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_PORT=29500

echo "=================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
if [ "$USE_HF_MODEL" = true ]; then
    echo "Model: $HF_MODEL"
else
    echo "Model: $MODEL_SIZE (from scratch)"
fi
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS * SLURM_NNODES * SLURM_GPUS_PER_NODE))"
echo "Checkpoint dir: $CHECKPOINT_DIR"
echo "=================================================="

# ============================================================================
# Build command
# ============================================================================

CMD="train_llm.py"

if [ "$USE_HF_MODEL" = true ]; then
    CMD="$CMD --hf-model $HF_MODEL"
else
    CMD="$CMD --from-scratch --model-size $MODEL_SIZE"
fi

CMD="$CMD --batch-size $BATCH_SIZE"
CMD="$CMD --grad-accum-steps $GRAD_ACCUM_STEPS"
CMD="$CMD --max-steps $MAX_STEPS"
CMD="$CMD --seq-len $SEQ_LEN"
CMD="$CMD --max-lr $MAX_LR"
CMD="$CMD --min-lr $MIN_LR"
CMD="$CMD --warmup-steps $WARMUP_STEPS"
CMD="$CMD --checkpoint-dir $CHECKPOINT_DIR"
CMD="$CMD --mixed-precision"
CMD="$CMD --use-flash-attn"

# ============================================================================
# Launch
# ============================================================================

srun torchrun \
    --nnodes $SLURM_NNODES \
    --nproc_per_node $SLURM_GPUS_PER_NODE \
    --rdzv_id $SLURM_JOB_ID \
    --rdzv_backend c10d \
    --rdzv_endpoint $MASTER_ADDR:$MASTER_PORT \
    $CMD

echo "Training completed at $(date)"

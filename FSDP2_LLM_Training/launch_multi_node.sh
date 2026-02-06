#!/bin/bash
# Multi-node training launcher (without SLURM)
# Run this script on each node with appropriate NODE_RANK

# ============================================================================
# Configuration - MUST SET THESE
# ============================================================================

# Cluster configuration
MASTER_ADDR="192.168.1.100"          # IP of the master node
MASTER_PORT=29500
NUM_NODES=4
NODE_RANK=${NODE_RANK:-0}            # Set this differently on each node (0, 1, 2, 3)
GPUS_PER_NODE=8

# Model configuration
USE_HF_MODEL=true
HF_MODEL="meta-llama/Llama-2-70b-hf"
MODEL_SIZE="70B"                     # Used if USE_HF_MODEL=false

# Training configuration
BATCH_SIZE=2
GRAD_ACCUM_STEPS=8
MAX_STEPS=100000

# Checkpointing (must be accessible from all nodes - use NFS/shared storage)
CHECKPOINT_DIR="/shared/checkpoints/llm_$(date +%Y%m%d_%H%M%S)"

# ============================================================================
# Info
# ============================================================================

echo "=================================================="
echo "Multi-Node Training Launcher"
echo "=================================================="
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Node rank: $NODE_RANK / $NUM_NODES"
echo "GPUs per node: $GPUS_PER_NODE"
if [ "$USE_HF_MODEL" = true ]; then
    echo "Model: $HF_MODEL"
else
    echo "Model: $MODEL_SIZE"
fi
echo "Total GPUs: $((NUM_NODES * GPUS_PER_NODE))"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_NODES * GPUS_PER_NODE))"
echo "=================================================="

# ============================================================================
# Environment
# ============================================================================

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0

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
CMD="$CMD --checkpoint-dir $CHECKPOINT_DIR"
CMD="$CMD --mixed-precision"

# ============================================================================
# Launch
# ============================================================================

torchrun \
    --nnodes $NUM_NODES \
    --nproc_per_node $GPUS_PER_NODE \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
    $CMD

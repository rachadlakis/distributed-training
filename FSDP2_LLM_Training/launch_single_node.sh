#!/bin/bash
# Single-node multi-GPU training launcher
# Usage: ./launch_single_node.sh [options]

set -e

# ============================================================================
# Configuration - Modify these as needed
# ============================================================================

# Choose one: Hugging Face model OR train from scratch
USE_HF_MODEL=true                    # Set to false for training from scratch
HF_MODEL="meta-llama/Llama-2-7b-hf"  # Options: meta-llama/Llama-2-7b-hf, mistralai/Mistral-7B-v0.1, etc.
MODEL_SIZE="7B"                      # Used if training from scratch: 350M, 1B, 3B, 7B, 13B, 30B, 70B

# Hardware
NUM_GPUS=8                           # Number of GPUs to use

# Training hyperparameters
BATCH_SIZE=4
GRAD_ACCUM_STEPS=2
MAX_STEPS=100000
SEQ_LEN=2048

# Optimizer
MAX_LR=3e-4
MIN_LR=3e-5
WARMUP_STEPS=2000
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0

# System
CHECKPOINT_DIR="checkpoints/$(date +%Y%m%d_%H%M%S)"
LOG_INTERVAL=10
CHECKPOINT_INTERVAL=1000

# ============================================================================
# Setup
# ============================================================================

echo "=================================================="
echo "LLM Training - Single Node"
echo "=================================================="
echo "GPUs: $NUM_GPUS"
if [ "$USE_HF_MODEL" = true ]; then
    echo "Model: $HF_MODEL (from Hugging Face)"
else
    echo "Model: $MODEL_SIZE (training from scratch)"
fi
echo "Batch size per GPU: $BATCH_SIZE"
echo "Gradient accumulation: $GRAD_ACCUM_STEPS"
echo "Effective batch size: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_GPUS))"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "=================================================="

# Create checkpoint directory
mkdir -p "$CHECKPOINT_DIR"

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
CMD="$CMD --weight-decay $WEIGHT_DECAY"
CMD="$CMD --grad-clip $GRAD_CLIP"
CMD="$CMD --checkpoint-dir $CHECKPOINT_DIR"
CMD="$CMD --checkpoint-interval $CHECKPOINT_INTERVAL"
CMD="$CMD --log-interval $LOG_INTERVAL"
CMD="$CMD --mixed-precision"
CMD="$CMD --use-flash-attn"

# ============================================================================
# Launch
# ============================================================================

echo "Launching training..."
echo ""

torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    $CMD

echo ""
echo "Training completed!"
echo "Checkpoints saved in: $CHECKPOINT_DIR"

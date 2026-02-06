#!/bin/bash
# Example: Fine-tune Llama 3 8B on 8 GPUs

set -e

echo "Fine-tuning Llama 3 8B on 8 GPUs"

torchrun \
    --standalone \
    --nproc_per_node=8 \
    train_llm.py \
    --hf-model meta-llama/Meta-Llama-3-8B \
    --batch-size 4 \
    --grad-accum-steps 2 \
    --seq-len 2048 \
    --max-lr 2e-5 \
    --min-lr 2e-6 \
    --warmup-steps 100 \
    --max-steps 10000 \
    --checkpoint-dir checkpoints/llama3-8b-finetuned \
    --mixed-precision \
    --use-flash-attn

echo "Training completed!"

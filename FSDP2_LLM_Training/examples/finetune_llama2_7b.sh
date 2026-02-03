#!/bin/bash
# Example: Fine-tune Llama 2 7B on 8 GPUs
# This script demonstrates fine-tuning a Llama 2 model

set -e

echo "Fine-tuning Llama 2 7B on 8 GPUs"

torchrun \
    --standalone \
    --nproc_per_node=8 \
    train_llm.py \
    --hf-model meta-llama/Llama-2-7b-hf \
    --batch-size 4 \
    --grad-accum-steps 2 \
    --seq-len 2048 \
    --max-lr 2e-5 \
    --min-lr 2e-6 \
    --warmup-steps 100 \
    --max-steps 10000 \
    --weight-decay 0.01 \
    --grad-clip 1.0 \
    --checkpoint-dir checkpoints/llama2-7b-finetuned \
    --checkpoint-interval 500 \
    --log-interval 10 \
    --mixed-precision \
    --use-flash-attn

echo "Training completed!"

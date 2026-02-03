#!/bin/bash
# Example: Fine-tune Mistral 7B on 8 GPUs

set -e

echo "Fine-tuning Mistral 7B on 8 GPUs"

torchrun \
    --standalone \
    --nproc_per_node=8 \
    train_llm.py \
    --hf-model mistralai/Mistral-7B-v0.1 \
    --batch-size 4 \
    --grad-accum-steps 2 \
    --seq-len 4096 \
    --max-lr 2e-5 \
    --min-lr 2e-6 \
    --warmup-steps 100 \
    --max-steps 10000 \
    --checkpoint-dir checkpoints/mistral-7b-finetuned \
    --mixed-precision \
    --use-flash-attn

echo "Training completed!"

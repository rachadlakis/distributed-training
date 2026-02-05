#!/bin/bash
# Example: Train a 7B model from scratch on 8 GPUs

set -e

echo "Training 7B model from scratch on 8 GPUs"

torchrun \
    --standalone \
    --nproc_per_node=8 \
    train_llm.py \
    --from-scratch \
    --model-size 7B \
    --vocab-size 32000 \
    --seq-len 2048 \
    --batch-size 4 \
    --grad-accum-steps 2 \
    --max-lr 3e-4 \
    --min-lr 3e-5 \
    --warmup-steps 2000 \
    --max-steps 100000 \
    --weight-decay 0.1 \
    --grad-clip 1.0 \
    --checkpoint-dir checkpoints/7b-from-scratch \
    --checkpoint-interval 1000 \
    --log-interval 10 \
    --mixed-precision

echo "Training completed!"

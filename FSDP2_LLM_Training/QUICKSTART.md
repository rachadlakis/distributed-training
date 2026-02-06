# Quick Start Guide

Get started with LLM training in 5 minutes!

## Prerequisites

- NVIDIA GPU(s) with CUDA 12.1+
- Python 3.10+
- At least 40GB disk space for model weights

## Step 1: Install Dependencies

```bash
# Install PyTorch
pip install torch>=2.5.0

# Install other dependencies
pip install -r requirements.txt
```

**For best performance (optional but recommended):**
```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
```

## Step 2: Authenticate with Hugging Face (if using gated models)

Some models like Llama 2 require authentication:

```bash
huggingface-cli login
```

Get your token from: https://huggingface.co/settings/tokens

Request access to gated models:
- Llama 2: https://huggingface.co/meta-llama/Llama-2-7b-hf
- Llama 3: https://huggingface.co/meta-llama/Meta-Llama-3-8B

## Step 3: Choose Your Scenario

### Scenario A: Fine-tune Llama 2 7B (Easiest)

Perfect for getting started quickly:

```bash
chmod +x examples/finetune_llama2_7b.sh
./examples/finetune_llama2_7b.sh
```

Or manually:
```bash
torchrun --nproc_per_node 8 train_llm.py \
    --hf-model meta-llama/Llama-2-7b-hf \
    --batch-size 4 \
    --grad-accum-steps 2 \
    --max-steps 1000 \
    --mixed-precision
```

### Scenario B: Fine-tune Mistral 7B

Great open-source alternative (no authentication needed):

```bash
chmod +x examples/finetune_mistral_7b.sh
./examples/finetune_mistral_7b.sh
```

### Scenario C: Train from Scratch

Build your own model:

```bash
chmod +x examples/train_from_scratch_7b.sh
./examples/train_from_scratch_7b.sh
```

## Step 4: Monitor Training

Training will output logs like:

```
==============================================================
Training Configuration
==============================================================
World size: 8 GPUs
Model: meta-llama/Llama-2-7b-hf (from Hugging Face)
Batch size per GPU: 4
Gradient accumulation: 2
Effective batch size: 64
Mixed precision: True
==============================================================

Loading model from Hugging Face: meta-llama/Llama-2-7b-hf
Model loaded: vocab_size=32000
Initializing new model...
Starting training...

Step     10 | Loss: 2.1234 | LR: 2.00e-06 | Tok/s:    50000 | Time: 1.23s
Step     20 | Loss: 1.9876 | LR: 4.00e-06 | Tok/s:    51000 | Time: 1.21s
...
```

## Step 5: Resume Training

Checkpoints are automatically saved. To resume:

```bash
# Just run the same command again - it will auto-resume from latest checkpoint
torchrun --nproc_per_node 8 train_llm.py \
    --hf-model meta-llama/Llama-2-7b-hf \
    --checkpoint-dir checkpoints/llama2-7b-finetuned \
    --mixed-precision
```

## Common Commands

### Check GPU Status
```bash
nvidia-smi
```

### Monitor GPU Usage in Real-time
```bash
watch -n 1 nvidia-smi
```

### List Checkpoints
```bash
ls -lh checkpoints/
```

### Test on Fewer GPUs
```bash
# Use 4 GPUs instead of 8
torchrun --nproc_per_node 4 train_llm.py \
    --hf-model meta-llama/Llama-2-7b-hf \
    --batch-size 2 \
    --mixed-precision
```

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
--batch-size 2  # or 1
```

Or increase gradient accumulation:
```bash
--grad-accum-steps 4  # or 8
```

### Model Download Issues

Models are downloaded to `~/.cache/huggingface/`. Make sure you have enough space:
```bash
df -h ~/.cache/huggingface/
```

### Permission Denied for Scripts

Make scripts executable:
```bash
chmod +x launch_single_node.sh
chmod +x examples/*.sh
```

## What's Next?

1. **Add your own data**: Replace `SimpleDataLoader` with your dataset
2. **Customize hyperparameters**: Edit launch scripts
3. **Scale to multiple nodes**: Use `launch_slurm.sh` or `launch_multi_node.sh`
4. **Add logging**: Integrate TensorBoard or Weights & Biases

See the main [README.md](README.md) for detailed documentation.

## Example Configurations

### Small Model (Testing)
```bash
torchrun --nproc_per_node 2 train_llm.py \
    --from-scratch \
    --model-size 350M \
    --batch-size 8 \
    --max-steps 100
```

### Medium Model (Single Node)
```bash
torchrun --nproc_per_node 8 train_llm.py \
    --hf-model meta-llama/Llama-2-7b-hf \
    --batch-size 4 \
    --grad-accum-steps 2
```

### Large Model (Multi-Node)
```bash
# See launch_slurm.sh or launch_multi_node.sh
sbatch launch_slurm.sh
```

## Help

For more information:
```bash
python train_llm.py --help
```

## Support

- Documentation: [README.md](README.md)
- PyTorch FSDP: https://pytorch.org/docs/stable/fsdp.html
- Hugging Face: https://huggingface.co/docs

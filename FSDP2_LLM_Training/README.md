# FSDP2 LLM Training

Complete solution for training and fine-tuning large language models using PyTorch FSDP2 (Fully Sharded Data Parallel v2) on single or multi-node GPU clusters.

## Features

- **Hugging Face Integration**: Fine-tune any model from Hugging Face (Llama 2, Mistral, etc.)
- **Train from Scratch**: Custom transformer architecture with modern features
- **Multi-Node Training**: Scale to hundreds of GPUs across multiple nodes
- **FSDP2 Sharding**: Efficient memory usage for models up to 70B+ parameters
- **Mixed Precision**: BF16 training for faster computation
- **Checkpointing**: Automatic save/resume with distributed checkpoints
- **Flash Attention**: Optimized attention mechanism
- **Production Ready**: Includes SLURM support and monitoring

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# For latest PyTorch (recommended)
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121

# For Hugging Face models requiring authentication
huggingface-cli login
```

### 2. Fine-tune a Hugging Face Model

Fine-tune Llama 2 7B on a single node with 8 GPUs:

```bash
chmod +x launch_single_node.sh
./launch_single_node.sh
```

Or manually:

```bash
torchrun --nproc_per_node 8 train_llm.py \
    --hf-model meta-llama/Llama-2-7b-hf \
    --batch-size 4 \
    --grad-accum-steps 2 \
    --mixed-precision
```

### 3. Train from Scratch

Train a custom 7B model from scratch:

```bash
torchrun --nproc_per_node 8 train_llm.py \
    --from-scratch \
    --model-size 7B \
    --batch-size 4 \
    --grad-accum-steps 2 \
    --mixed-precision
```

## Supported Models

### Hugging Face Models

The script supports any causal language model from Hugging Face. Popular choices:

| Model | HF Model Name | Recommended GPUs | Batch Size |
|-------|---------------|------------------|------------|
| Llama 2 7B | `meta-llama/Llama-2-7b-hf` | 8 | 4 |
| Llama 2 13B | `meta-llama/Llama-2-13b-hf` | 8 | 2 |
| Llama 2 70B | `meta-llama/Llama-2-70b-hf` | 32-64 | 1-2 |
| Llama 3 8B | `meta-llama/Meta-Llama-3-8B` | 8 | 4 |
| Llama 3 70B | `meta-llama/Meta-Llama-3-70B` | 32-64 | 1-2 |
| Mistral 7B | `mistralai/Mistral-7B-v0.1` | 8 | 4 |
| Mixtral 8x7B | `mistralai/Mixtral-8x7B-v0.1` | 16-32 | 2 |
| Falcon 7B | `tiiuae/falcon-7b` | 8 | 4 |
| Falcon 40B | `tiiuae/falcon-40b` | 32 | 2 |
| Phi-2 | `microsoft/phi-2` | 2 | 8 |
| CodeLlama 7B | `codellama/CodeLlama-7b-hf` | 8 | 4 |

### Custom Models (From Scratch)

| Size | Parameters | Layers | Hidden Dim | Heads | Min GPUs |
|------|-----------|--------|------------|-------|----------|
| 350M | ~350M | 24 | 1024 | 16 | 1 |
| 1B | ~1B | 24 | 2048 | 16 | 2 |
| 3B | ~3B | 28 | 3072 | 24 | 4 |
| 7B | ~7B | 32 | 4096 | 32 | 8 |
| 13B | ~13B | 40 | 5120 | 40 | 8 |
| 30B | ~30B | 60 | 6656 | 52 | 16 |
| 70B | ~70B | 80 | 8192 | 64 | 32 |

## Usage Examples

### Example 1: Fine-tune Llama 2 7B (Single Node)

```bash
torchrun --nproc_per_node 8 train_llm.py \
    --hf-model meta-llama/Llama-2-7b-hf \
    --batch-size 4 \
    --grad-accum-steps 2 \
    --max-lr 2e-5 \
    --warmup-steps 100 \
    --max-steps 10000 \
    --checkpoint-dir checkpoints/llama2-7b-finetuned \
    --mixed-precision
```

### Example 2: Fine-tune Mistral 7B with Custom Sequence Length

```bash
torchrun --nproc_per_node 8 train_llm.py \
    --hf-model mistralai/Mistral-7B-v0.1 \
    --seq-len 4096 \
    --batch-size 2 \
    --grad-accum-steps 4 \
    --mixed-precision
```

### Example 3: Train 13B Model from Scratch

```bash
torchrun --nproc_per_node 8 train_llm.py \
    --from-scratch \
    --model-size 13B \
    --vocab-size 32000 \
    --seq-len 2048 \
    --batch-size 2 \
    --grad-accum-steps 4 \
    --max-lr 3e-4 \
    --mixed-precision
```

### Example 4: Multi-Node Training (SLURM)

```bash
sbatch launch_slurm.sh
```

Or configure `launch_slurm.sh` with your settings:
```bash
USE_HF_MODEL=true
HF_MODEL="meta-llama/Llama-2-70b-hf"
BATCH_SIZE=2
GRAD_ACCUM_STEPS=8
```

### Example 5: Multi-Node Training (Manual)

On master node (NODE_RANK=0):
```bash
export NODE_RANK=0
./launch_multi_node.sh
```

On worker nodes (NODE_RANK=1,2,3...):
```bash
export NODE_RANK=1  # Change for each node
./launch_multi_node.sh
```

## Command-Line Arguments

### Model Selection (Choose One)

- `--hf-model MODEL_NAME`: Use Hugging Face model (e.g., `meta-llama/Llama-2-7b-hf`)
- `--from-scratch`: Train custom model from scratch

### Model Configuration (for --from-scratch)

- `--model-size {350M,1B,3B,7B,13B,30B,70B}`: Model size [default: 7B]
- `--vocab-size INT`: Vocabulary size [default: 32000]
- `--seq-len INT`: Sequence length [default: 2048]

### Training Configuration

- `--batch-size INT`: Batch size per GPU [default: 4]
- `--grad-accum-steps INT`: Gradient accumulation steps [default: 1]
- `--max-steps INT`: Maximum training steps [default: 100000]
- `--num-epochs INT`: Number of epochs [default: 1]
- `--steps-per-epoch INT`: Steps per epoch [default: 1000]

### Optimizer Configuration

- `--max-lr FLOAT`: Maximum learning rate [default: 3e-4]
- `--min-lr FLOAT`: Minimum learning rate [default: 3e-5]
- `--warmup-steps INT`: Warmup steps [default: 2000]
- `--weight-decay FLOAT`: Weight decay [default: 0.1]
- `--grad-clip FLOAT`: Gradient clipping [default: 1.0]

### FSDP Configuration

- `--mixed-precision`: Enable BF16 mixed precision [default: True]
- `--use-flash-attn`: Use Flash Attention 2 [default: True]

### Checkpointing

- `--checkpoint-dir PATH`: Checkpoint directory [default: checkpoints]
- `--checkpoint-interval INT`: Save interval in steps [default: 1000]
- `--log-interval INT`: Logging interval in steps [default: 10]

## File Structure

```
FSDP2_LLM_Training/
├── train_llm.py              # Main training script
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── launch_single_node.sh      # Single-node launcher
├── launch_slurm.sh           # SLURM multi-node launcher
├── launch_multi_node.sh      # Manual multi-node launcher
└── examples/                  # Example configurations (optional)
```

## Memory Requirements

Approximate GPU memory per model (with mixed precision):

| Model | 8 GPUs | 16 GPUs | 32 GPUs | 64 GPUs |
|-------|--------|---------|---------|---------|
| 7B | ~12 GB | ~8 GB | ~6 GB | ~4 GB |
| 13B | ~20 GB | ~12 GB | ~8 GB | ~6 GB |
| 30B | ~40 GB | ~24 GB | ~14 GB | ~10 GB |
| 70B | OOM | ~40 GB | ~24 GB | ~14 GB |

## Customization

### Using Your Own Dataset

Replace the `SimpleDataLoader` in `train_llm.py`:

```python
from torch.utils.data import DataLoader, DistributedSampler
from datasets import load_dataset

# Load dataset
dataset = load_dataset("your-dataset")

# Create sampler
sampler = DistributedSampler(
    dataset,
    num_replicas=world_size,
    rank=rank,
    shuffle=True,
)

# Create dataloader
train_loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    sampler=sampler,
    num_workers=4,
    pin_memory=True,
)
```

### Adding Logging

**TensorBoard:**
```python
from torch.utils.tensorboard import SummaryWriter

if is_main:
    writer = SummaryWriter(log_dir="runs")
    writer.add_scalar("Loss/train", loss.item(), total_steps)
```

**Weights & Biases:**
```python
import wandb

if is_main:
    wandb.init(project="llm-training")
    wandb.log({"loss": loss.item(), "lr": lr})
```

## Troubleshooting

### Out of Memory (OOM)

1. Reduce `--batch-size`
2. Increase `--grad-accum-steps`
3. Reduce `--seq-len`
4. Use more GPUs
5. Ensure `--mixed-precision` is enabled

### Slow Training

1. Enable `--mixed-precision`
2. Enable `--use-flash-attn`
3. Increase batch size if memory allows
4. Check GPU utilization: `nvidia-smi dmon`
5. Verify network bandwidth between nodes

### Hugging Face Authentication

Some models require authentication:
```bash
huggingface-cli login
# Enter your access token from https://huggingface.co/settings/tokens
```

### NCCL Errors (Multi-Node)

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0  # Enable InfiniBand
export NCCL_SOCKET_IFNAME=eth0  # Or your network interface
```

## Performance Tips

1. **Use BF16**: Always enable `--mixed-precision`
2. **Maximize Batch Size**: Use largest batch size that fits in memory
3. **Gradient Accumulation**: Simulate larger batches without OOM
4. **Flash Attention**: Enable with `--use-flash-attn`
5. **Network**: Use InfiniBand for multi-node training
6. **Storage**: Use fast shared storage (Lustre, NFS) for checkpoints

## Best Practices

### For Fine-tuning

- Use lower learning rates (1e-5 to 1e-4)
- Shorter warmup (100-500 steps)
- Smaller number of steps

```bash
torchrun --nproc_per_node 8 train_llm.py \
    --hf-model meta-llama/Llama-2-7b-hf \
    --max-lr 2e-5 \
    --warmup-steps 100 \
    --max-steps 5000
```

### For Training from Scratch

- Higher learning rates (3e-4 to 1e-3)
- Longer warmup (2000-4000 steps)
- Many more steps

```bash
torchrun --nproc_per_node 8 train_llm.py \
    --from-scratch \
    --model-size 7B \
    --max-lr 3e-4 \
    --warmup-steps 2000 \
    --max-steps 100000
```

## Architecture Features

This implementation includes:

- **Grouped Query Attention (GQA)**: Memory-efficient attention
- **RoPE (Rotary Position Embeddings)**: Better position encoding
- **SwiGLU Activation**: Superior to ReLU/GELU
- **RMSNorm**: More stable than LayerNorm
- **Flash Attention**: Optimized attention computation
- **Pre-normalization**: Improved training stability

## License

This code is provided for educational and research purposes.

## Support

For issues, questions, or contributions:
- Check existing issues and documentation
- Review PyTorch FSDP documentation
- Consult Hugging Face model documentation

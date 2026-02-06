"""
Comprehensive FSDP2 Training Script for Large Language Models
Supports:
- Training from scratch with custom architecture
- Fine-tuning models from Hugging Face (Llama, Mistral, etc.)
- Multi-node, multi-GPU distributed training
- Mixed precision, gradient accumulation, checkpointing

Usage:
    Train from scratch:
        torchrun --nproc_per_node 8 train_llm.py --model-size 7B --from-scratch

    Fine-tune Llama 2:
        torchrun --nproc_per_node 8 train_llm.py --hf-model meta-llama/Llama-2-7b-hf

    Fine-tune Mistral:
        torchrun --nproc_per_node 8 train_llm.py --hf-model mistralai/Mistral-7B-v0.1

    Multi-node:
        torchrun --nnodes 4 --nproc_per_node 8 --rdzv_id=JOB --rdzv_backend=c10d \
                 --rdzv_endpoint=MASTER:29500 train_llm.py --hf-model meta-llama/Llama-2-70b-hf
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.checkpoint.state_dict import (
    _init_optim_state,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, FSDPModule
from torch.distributed.tensor import distribute_tensor, DTensor


# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """LLM Model Configuration"""
    vocab_size: int = 32000
    max_seq_len: int = 2048
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    hidden_dim: Optional[int] = None
    dropout_p: float = 0.0
    use_bias: bool = False
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5

    def __post_init__(self):
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        if self.hidden_dim is None:
            self.hidden_dim = 4 * self.dim

    @classmethod
    def from_name(cls, name: str) -> "ModelConfig":
        """Create model config from predefined sizes"""
        configs = {
            "350M": cls(dim=1024, n_layers=24, n_heads=16, n_kv_heads=16),
            "1B": cls(dim=2048, n_layers=24, n_heads=16, n_kv_heads=16),
            "3B": cls(dim=3072, n_layers=28, n_heads=24, n_kv_heads=8),
            "7B": cls(dim=4096, n_layers=32, n_heads=32, n_kv_heads=8),
            "13B": cls(dim=5120, n_layers=40, n_heads=40, n_kv_heads=8),
            "30B": cls(dim=6656, n_layers=60, n_heads=52, n_kv_heads=8),
            "70B": cls(dim=8192, n_layers=80, n_heads=64, n_kv_heads=8),
        }
        return configs.get(name, cls())


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight

    def reset_parameters(self):
        nn.init.ones_(self.weight)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim: int, max_seq_len: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x, seq_len: int):
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


def apply_rotary_emb(xq, xk, cos, sin):
    """Apply rotary embeddings"""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    xq_out = (xq * cos) + (rotate_half(xq) * sin)
    xk_out = (xk * cos) + (rotate_half(xk) * sin)
    return xq_out, xk_out


class GroupedQueryAttention(nn.Module):
    """Multi-Head Attention with GQA"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=config.use_bias)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=config.use_bias)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=config.use_bias)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=config.use_bias)
        self.dropout_p = config.dropout_p
        self.rotary_emb = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(self, x):
        bsz, seq_len, _ = x.size()
        xq = self.wq(x).view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        cos, sin = self.rotary_emb(xq, seq_len)
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        if self.n_rep > 1:
            xk = xk.repeat_interleave(self.n_rep, dim=2)
            xv = xv.repeat_interleave(self.n_rep, dim=2)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        output = F.scaled_dot_product_attention(
            xq, xk, xv, dropout_p=self.dropout_p if self.training else 0.0, is_causal=True
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.wo(output)

    def reset_parameters(self):
        for m in [self.wq, self.wk, self.wv, self.wo]:
            nn.init.xavier_uniform_(m.weight)


class FeedForward(nn.Module):
    """SwiGLU Feed-Forward"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.hidden_dim, bias=config.use_bias)
        self.w2 = nn.Linear(config.hidden_dim, config.dim, bias=config.use_bias)
        self.w3 = nn.Linear(config.dim, config.hidden_dim, bias=config.use_bias)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

    def reset_parameters(self):
        for m in [self.w1, self.w2, self.w3]:
            nn.init.xavier_uniform_(m.weight)


class TransformerBlock(nn.Module):
    """Transformer decoder block"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention = GroupedQueryAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        return h + self.feed_forward(self.ffn_norm(h))

    def reset_parameters(self):
        self.attention_norm.reset_parameters()
        self.attention.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.feed_forward.reset_parameters()


class LLMTransformer(nn.Module):
    """Custom LLM Transformer"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(self, tokens, targets=None):
        h = self.tok_embeddings(tokens)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        logits = self.output(h)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-1)
        return logits, loss

    def reset_parameters(self):
        nn.init.normal_(self.tok_embeddings.weight, std=0.02)
        for layer in self.layers:
            layer.reset_parameters()
        self.norm.reset_parameters()
        nn.init.normal_(self.output.weight, std=0.02)


# ============================================================================
# Hugging Face Model Wrapper
# ============================================================================

class HuggingFaceModelWrapper(nn.Module):
    """Wrapper for Hugging Face models to provide unified interface"""
    def __init__(self, model_name: str, use_flash_attention: bool = True):
        super().__init__()
        try:
            from transformers import AutoModelForCausalLM, AutoConfig
        except ImportError:
            raise ImportError("transformers library required. Install: pip install transformers")

        print(f"Loading model from Hugging Face: {model_name}")

        # Configure model loading
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

        # Enable flash attention if available
        if use_flash_attention:
            config.use_flash_attention_2 = True

        # Load model with meta device for FSDP
        with torch.device("meta"):
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config=config,
                torch_dtype=torch.float32,
                trust_remote_code=True,
            )

        self.vocab_size = config.vocab_size
        print(f"Model loaded: vocab_size={self.vocab_size}")

    def forward(self, tokens, targets=None):
        outputs = self.model(input_ids=tokens, labels=targets)

        if targets is not None:
            return outputs.logits, outputs.loss
        return outputs.logits, None

    def reset_parameters(self):
        # HF models are already initialized
        pass


# ============================================================================
# Checkpointing
# ============================================================================

class Checkpointer:
    """Handles distributed checkpointing"""
    def __init__(self, checkpoint_dir: str, use_dcp_api: bool = True):
        self.checkpoint_dir = checkpoint_dir
        self.use_dcp_api = use_dcp_api
        self.last_checkpoint = self._get_latest_checkpoint()

    def _get_latest_checkpoint(self):
        if not os.path.exists(self.checkpoint_dir):
            return None
        checkpoints = []
        for name in os.listdir(self.checkpoint_dir):
            path = os.path.join(self.checkpoint_dir, name)
            if os.path.isdir(path) and name.startswith("checkpoint_"):
                try:
                    step = int(name.split("_")[1])
                    checkpoints.append((step, path))
                except ValueError:
                    pass
        if not checkpoints:
            return None
        checkpoints.sort(reverse=True)
        return checkpoints[0][1]

    def save(self, model: FSDPModule, optimizer: torch.optim.Optimizer, step: int, metrics: Dict):
        checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_{step}")
        if torch.distributed.get_rank() == 0:
            os.makedirs(checkpoint_path, exist_ok=True)
        torch.distributed.barrier()

        model_state = self._get_full_state_dict(model, is_model=True)
        if torch.distributed.get_rank() == 0:
            torch.save(model_state, os.path.join(checkpoint_path, "model.pt"))

        optim_state = self._get_full_state_dict(optimizer, is_model=False, model=model)
        if torch.distributed.get_rank() == 0:
            torch.save(optim_state, os.path.join(checkpoint_path, "optimizer.pt"))
            with open(os.path.join(checkpoint_path, "metadata.json"), "w") as f:
                json.dump(metrics, f, indent=2)
        torch.distributed.barrier()

    def load(self, model: FSDPModule, optimizer: Optional[torch.optim.Optimizer] = None):
        if self.last_checkpoint is None:
            return None

        model_path = os.path.join(self.last_checkpoint, "model.pt")
        state_dict = torch.load(model_path, mmap=True, weights_only=True, map_location="cpu")

        if self.use_dcp_api:
            set_model_state_dict(model, state_dict, options=StateDictOptions(
                full_state_dict=True, broadcast_from_rank0=True))
        else:
            self._load_sharded_state_dict(model, state_dict)

        if optimizer is not None:
            optim_path = os.path.join(self.last_checkpoint, "optimizer.pt")
            optim_state = torch.load(optim_path, mmap=True, weights_only=True, map_location="cpu")
            if self.use_dcp_api:
                set_optimizer_state_dict(model, optimizer, optim_state, options=StateDictOptions(
                    full_state_dict=True, broadcast_from_rank0=True))

        metadata_path = os.path.join(self.last_checkpoint, "metadata.json")
        if os.path.exists(metadata_path) and torch.distributed.get_rank() == 0:
            with open(metadata_path) as f:
                return json.load(f)
        return None

    def _get_full_state_dict(self, obj, is_model=True, model=None):
        if is_model:
            return get_model_state_dict(obj, options=StateDictOptions(
                full_state_dict=True, cpu_offload=True)) if self.use_dcp_api else self._get_full_model_state(obj)
        return get_optimizer_state_dict(model, obj, options=StateDictOptions(
            full_state_dict=True, cpu_offload=True)) if self.use_dcp_api else {}

    def _get_full_model_state(self, model):
        sharded_sd = model.state_dict()
        cpu_sd = {}
        for name, param in sharded_sd.items():
            full_param = param.full_tensor()
            if torch.distributed.get_rank() == 0:
                cpu_sd[name] = full_param.cpu()
            else:
                del full_param
        return cpu_sd

    def _load_sharded_state_dict(self, model, full_sd):
        meta_sd = model.state_dict()
        sharded_sd = {}
        for name, tensor in full_sd.items():
            meta_param = meta_sd.get(name)
            sharded_sd[name] = nn.Parameter(distribute_tensor(
                tensor, meta_param.device_mesh, meta_param.placements))
        model.load_state_dict(sharded_sd, strict=False, assign=True)


# ============================================================================
# Data Loading
# ============================================================================

class SimpleDataLoader:
    """Simple data loader - replace with real data pipeline"""
    def __init__(self, vocab_size, seq_len, batch_size, num_batches, device):
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.device = device
        self.current_batch = 0

    def __iter__(self):
        self.current_batch = 0
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        self.current_batch += 1
        tokens = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        targets = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_len), device=self.device)
        return tokens, targets

    def __len__(self):
        return self.num_batches


# ============================================================================
# Training
# ============================================================================

def setup_distributed() -> Tuple[int, int, int, torch.device]:
    """Initialize distributed training"""
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend, device_id=device)

    return rank, local_rank, world_size, device


def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine schedule with warmup"""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step > max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def create_model(args, is_main):
    """Create model from scratch or load from Hugging Face"""
    if args.hf_model:
        if is_main:
            print(f"Loading Hugging Face model: {args.hf_model}")
        model = HuggingFaceModelWrapper(args.hf_model, use_flash_attention=args.use_flash_attn)
    else:
        if is_main:
            print(f"Creating custom model from scratch: {args.model_size}")
        config = ModelConfig.from_name(args.model_size)
        if args.vocab_size:
            config.vocab_size = args.vocab_size
        if args.seq_len:
            config.max_seq_len = args.seq_len
        with torch.device("meta"):
            model = LLMTransformer(config)

    return model


def train(args):
    """Main training function"""
    rank, local_rank, world_size, device = setup_distributed()
    is_main = rank == 0

    if is_main:
        print(f"\n{'='*60}")
        print(f"Training Configuration")
        print(f"{'='*60}")
        print(f"World size: {world_size} GPUs")
        print(f"Model: {args.hf_model if args.hf_model else args.model_size}")
        print(f"Batch size per GPU: {args.batch_size}")
        print(f"Gradient accumulation: {args.grad_accum_steps}")
        print(f"Effective batch size: {args.batch_size * args.grad_accum_steps * world_size}")
        print(f"Mixed precision: {args.mixed_precision}")
        print(f"{'='*60}\n")

    # Create model
    model = create_model(args, is_main)

    # Setup FSDP
    fsdp_kwargs = {}
    if args.mixed_precision:
        fsdp_kwargs["mp_policy"] = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32)

    # Shard model layers
    if hasattr(model, 'layers'):
        for layer in model.layers:
            fully_shard(layer, **fsdp_kwargs)
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            fully_shard(layer, **fsdp_kwargs)

    fully_shard(model, **fsdp_kwargs)

    # Checkpointing
    checkpointer = Checkpointer(args.checkpoint_dir, use_dcp_api=True)

    if checkpointer.last_checkpoint is None:
        if is_main:
            print("Initializing new model...")
        model.to_empty(device=device)
        model.reset_parameters()
        start_step = 0
    else:
        if is_main:
            print(f"Resuming from checkpoint: {checkpointer.last_checkpoint}")
        metadata = checkpointer.load(model)
        start_step = metadata.get("step", 0) if metadata else 0

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )

    if checkpointer.last_checkpoint:
        checkpointer.load(model, optimizer)

    # Data loader
    train_loader = SimpleDataLoader(
        vocab_size=model.vocab_size,
        seq_len=args.seq_len or 2048,
        batch_size=args.batch_size,
        num_batches=args.steps_per_epoch,
        device=device,
    )

    # Training loop
    model.train()
    total_steps = start_step
    tokens_seen = 0

    if is_main:
        print("Starting training...\n")

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_start = time.time()

        for step, (tokens, targets) in enumerate(train_loader):
            step_start = time.time()

            _, loss = model(tokens, targets)
            loss = loss / args.grad_accum_steps
            loss.backward()

            if (step + 1) % args.grad_accum_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                lr = get_lr(total_steps, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

                optimizer.step()
                optimizer.zero_grad()
                total_steps += 1
                tokens_seen += args.batch_size * (args.seq_len or 2048) * args.grad_accum_steps * world_size

            epoch_loss += loss.item() * args.grad_accum_steps

            if total_steps % args.log_interval == 0 and is_main:
                step_time = time.time() - step_start
                toks_per_sec = (args.batch_size * (args.seq_len or 2048) * world_size) / step_time
                print(f"Step {total_steps:6d} | Loss: {loss.item() * args.grad_accum_steps:.4f} | "
                      f"LR: {lr:.2e} | Tok/s: {toks_per_sec:8.0f} | Time: {step_time:.2f}s")

            if total_steps % args.checkpoint_interval == 0 and total_steps > 0:
                if is_main:
                    print(f"\nSaving checkpoint at step {total_steps}...")
                checkpointer.save(model, optimizer, total_steps, {
                    "step": total_steps, "epoch": epoch, "loss": epoch_loss / (step + 1),
                    "lr": lr, "tokens_seen": tokens_seen})

            if total_steps >= args.max_steps:
                break

        if is_main:
            print(f"\nEpoch {epoch + 1} | Loss: {epoch_loss / len(train_loader):.4f} | "
                  f"Time: {time.time() - epoch_start:.2f}s\n")

        if total_steps >= args.max_steps:
            break

    # Final checkpoint
    if is_main:
        print("Saving final checkpoint...")
    checkpointer.save(model, optimizer, total_steps, {
        "step": total_steps, "loss": epoch_loss / len(train_loader), "tokens_seen": tokens_seen})

    torch.distributed.destroy_process_group()
    if is_main:
        print("\nTraining completed!")


def main():
    parser = argparse.ArgumentParser(description="Train LLM with FSDP2")

    # Model source
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument("--hf-model", type=str, help="Hugging Face model name (e.g., meta-llama/Llama-2-7b-hf)")
    model_group.add_argument("--from-scratch", action="store_true", help="Train from scratch")

    parser.add_argument("--model-size", type=str, default="7B",
                        choices=["350M", "1B", "3B", "7B", "13B", "30B", "70B"],
                        help="Model size (for from-scratch)")
    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--use-flash-attn", action="store_true", default=True)

    # Training
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=100000)

    # Optimizer
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # FSDP
    parser.add_argument("--mixed-precision", action="store_true", default=True)

    # Checkpointing
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=10)

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

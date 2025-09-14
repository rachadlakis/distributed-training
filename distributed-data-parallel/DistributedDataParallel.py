# ----------------------------
# pip install torch numpy
# ----------------------------

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# ----------------------------
# 1. Print PyTorch info
# ----------------------------
print(f"PyTorch Version: {torch.__version__}")
print("-" * 40)

# ----------------------------
# 2. Define a simple model
# ----------------------------
class SimpleModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)

    def forward(self, x, debug=False):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        if debug:
            # Print input/output shapes per rank for debugging
            print(f"[Rank {dist.get_rank()}] Input: {x.shape}, Output: {out.shape}")
        return out

# ----------------------------
# 3. Initialize Distributed Process Group
# ----------------------------
def setup_distributed():
    """
    Initializes the default process group for DDP.
    Uses environment variables set by torchrun:
      - LOCAL_RANK: GPU assigned to this process
      - WORLD_SIZE: total number of processes
    """
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return local_rank, device

# ----------------------------
# 4. Cleanup Distributed
# ----------------------------
def cleanup_distributed():
    dist.destroy_process_group()

# ----------------------------
# 5. Training Function
# ----------------------------
def train(input_size, output_size, batch_size, learning_rate, num_epochs):
    # Setup distributed environment
    local_rank, device = setup_distributed()
    world_size = dist.get_world_size()

    if local_rank == 0:
        print(f"Training on {world_size} GPUs (DDP)")

    # ----------------------------
    # 5a. Create synthetic dataset
    # ----------------------------
    num_samples = 10000
    inputs = torch.randn(num_samples, input_size)
    targets = torch.randint(0, output_size, (num_samples,))
    dataset = TensorDataset(inputs, targets)

    # DistributedSampler splits dataset across processes
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # ----------------------------
    # 5b. Model, Loss, Optimizer
    # ----------------------------
    model = SimpleModel(input_size, output_size).to(device)
    model = DDP(model, device_ids=[local_rank])  # Wrap model with DDP
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # ----------------------------
    # 5c. Training Loop
    # ----------------------------
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # reshuffle dataset for each epoch
        total_loss = 0.0

        for i, (batch_inputs, batch_targets) in enumerate(data_loader):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            # Optional debug for first batch of first epoch
            debug = (epoch == 0 and i == 0)
            outputs = model(batch_inputs, debug=debug)

            loss = criterion(outputs, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)

        # Only rank 0 prints to avoid clutter
        if local_rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {avg_loss:.4f}")

    # ----------------------------
    # 5d. Save Model (only rank 0)
    # ----------------------------
    if local_rank == 0:
        torch.save(model.module.state_dict(), "ddp_model.pth")
        print("✅ Model saved on rank 0")

    # Cleanup
    cleanup_distributed()

# ----------------------------
# 6. Main entry point
# ----------------------------
def main():
    # Hyperparameters
    input_size = 784
    output_size = 10
    batch_size = 256
    learning_rate = 0.01
    num_epochs = 20

    # Run training
    train(input_size, output_size, batch_size, learning_rate, num_epochs)

# ----------------------------
# 7. Run script
# ----------------------------
if __name__ == "__main__":
    main()

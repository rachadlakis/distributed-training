# pip install torch
# pip install numpy

# Imports and Initial Setup
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

print(f"PyTorch Version: {torch.__version__}")
print("-" * 30)

# Define a Simple Model
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
            print(f"\tInside the Model (rank {dist.get_rank()}): input size {x.size()}, output size {out.size()}")
        return out

# Training Loop (per process)
def train(rank, world_size, input_size, output_size, batch_size, learning_rate, num_epochs):
    # Setup distributed process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Data Preparation
    inputs = torch.randn(10000, input_size)
    targets = torch.randint(0, output_size, (10000,))
    dataset = TensorDataset(inputs, targets)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    # Model, Loss, Optimizer
    model = SimpleModel(input_size, output_size).to(device)
    model = DDP(model, device_ids=[rank])
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # Training Loop
    if rank == 0:
        print("\nStarting training with DDP...")

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # ensures each rank gets a new shuffle
        total_loss = 0
        for i, (batch_inputs, batch_targets) in enumerate(data_loader):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)

            debug = (epoch == 0 and i == 0)
            outputs = model(batch_inputs, debug=debug)

            if debug and rank == 0:
                print("Outside: input size", batch_inputs.size(), "output_size", outputs.size())

            loss = criterion(outputs, batch_targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        if rank == 0:  # only print from rank 0
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

    if rank == 0:
        torch.save(model.module.state_dict(), 'model_state_ddp.pth')
        print("\n✅ Model saved on rank 0")

    # Cleanup
    dist.destroy_process_group()

# Main entry point
def main():
    input_size = 784
    output_size = 10
    batch_size = 256
    learning_rate = 0.01
    num_epochs = 20

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        print(f"Found {world_size} GPUs.")
    else:
        world_size = 0
        print("No GPUs found. DDP requires GPUs.")
        return

    # Set environment variables for DDP
    os.environ['MASTER_ADDR'] = 'localhost'   # or 'localhost'
    os.environ['MASTER_PORT'] = '29500'       # any free port

    mp.spawn(
        train,
        args=(world_size, input_size, output_size, batch_size, learning_rate, num_epochs),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()

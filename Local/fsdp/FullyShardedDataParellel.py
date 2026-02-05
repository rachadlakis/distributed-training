import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from torch.distributed.fsdp.wrap import default_auto_wrap_policy

from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

# ------------------------------
# Simple Model
# ------------------------------
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

# ------------------------------
# Setup / Cleanup
# ------------------------------
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

# ------------------------------
# Data
# ------------------------------
def get_dataloader(input_size, output_size, batch_size=32):
    inputs = torch.randn(10000, input_size)
    targets = torch.randint(0, output_size, (10000,))
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader

# ------------------------------
# Training function
# ------------------------------
def train_simple(rank, world_size, input_size=784, output_size=10, batch_size=32, lr=0.01, epochs=5):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    train_loader = get_dataloader(input_size, output_size, batch_size)

    model = SimpleModel(input_size, output_size).to(device)
    # fsdp_model = FSDP(model, auto_wrap_policy=default_auto_wrap_policy)
    fsdp_model = FSDP(model, auto_wrap_policy=size_based_auto_wrap_policy)

    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fsdp_model.parameters(), lr=lr)

    for epoch in range(epochs):
        fsdp_model.train()
        running_loss = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = fsdp_model(x, debug=(epoch==0 and i==0))
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if rank == 0:
            print(f"[Rank {rank}] Epoch {epoch+1}, Avg Loss: {running_loss/len(train_loader):.4f}")

    if rank == 0:
        torch.save(fsdp_model.state_dict(), 'simplemodel_fsdp.pth')
        print("✅ SimpleModel saved on rank 0")

    cleanup()

# ------------------------------
# Main
# ------------------------------
def main_simple():
    world_size = torch.cuda.device_count()
    mp.spawn(train_simple, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main_simple()

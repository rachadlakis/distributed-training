import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import default_auto_wrap_policy

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
def get_cifar10_loader(batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    subset = Subset(dataset, range(int(0.1*len(dataset))))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True)
    return loader

# ------------------------------
# Training function
# ------------------------------
def train_resnet(rank, world_size, batch_size=32, lr=0.001, epochs=5):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    train_loader = get_cifar10_loader(batch_size)

    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 classes
    model.to(device)

    fsdp_model = FSDP(model, auto_wrap_policy=default_auto_wrap_policy)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(fsdp_model.parameters(), lr=lr)

    for epoch in range(epochs):
        fsdp_model.train()
        running_loss = 0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = fsdp_model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if rank == 0:
            print(f"[Rank {rank}] Epoch {epoch+1}, Avg Loss: {running_loss/len(train_loader):.4f}")

    if rank == 0:
        torch.save(fsdp_model.state_dict(), 'resnet50_fsdp.pth')
        print("✅ ResNet-50 saved on rank 0")

    cleanup()

# ------------------------------
# Main
# ------------------------------
def main_resnet():
    world_size = torch.cuda.device_count()
    mp.spawn(train_resnet, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main_resnet()

# Imports and Initial Setup
import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler, Subset, Dataset
from torchvision import transforms, models
from datasets import load_dataset
from torch.nn.parallel import DistributedDataParallel as DDP

print(f"PyTorch Version: {torch.__version__}")
print("-" * 30)

# Custom Dataset for CIFAR-10
class CIFAR10Dataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.data = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]["img"]
        label = self.data[idx]["label"]
        if self.transform:
            img = self.transform(img)
        return img, label

# Training Loop (per process)
def train(rank, world_size, batch_size, learning_rate, num_epochs):
    # Initialize DDP
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    # Load CIFAR-10 (10% of the dataset)
    dataset = load_dataset("cifar10")
    train_subset = Subset(dataset["train"], range(int(0.1 * len(dataset["train"]))))
    
    # Transformations for ResNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = CIFAR10Dataset(train_subset, transform=transform)

    # Distributed Sampler
    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    data_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    # Model, Loss, Optimizer
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 has 10 classes
    model = model.to(device)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if rank == 0:
        print("\nStarting ResNet-50 training with DDP on CIFAR-10 10% subset...")

    # Training loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)
        running_loss = 0.0
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if rank == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {running_loss/len(data_loader):.4f}")

    # Save model only on rank 0
    if rank == 0:
        torch.save(model.module.state_dict(), "resnet50_cifar10_ddp.pth")
        print("\n✅ Model saved on rank 0")

    # Cleanup
    dist.destroy_process_group()

# Main entry point
def main():
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 5

    if torch.cuda.is_available():
        world_size = torch.cuda.device_count()
        print(f"Found {world_size} GPUs for DDP training.")
    else:
        world_size = 0
        print("No GPUs found. DDP requires GPUs.")
        return

    mp.spawn(
        train,
        args=(world_size, batch_size, learning_rate, num_epochs),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main()

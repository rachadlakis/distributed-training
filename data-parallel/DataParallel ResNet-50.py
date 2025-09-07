import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from datasets import load_dataset

# Check for available GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} GPUs for training.")

# Load CIFAR-10 dataset from Hugging Face
dataset = load_dataset("cifar10")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet-50 expects 224x224 images
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Custom Dataset class
class CIFAR10Dataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data["img"])

    def __getitem__(self, idx):
        img = self.data["img"][idx]
        label = self.data["label"][idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Create datasets
train_dataset = CIFAR10Dataset(train_dataset, transform=transform)
test_dataset = CIFAR10Dataset(test_dataset, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Load pre-trained ResNet-50 and modify for CIFAR-10
model = models.resnet50(weights=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # Replace final layer for 10 classes

# Wrap the model with DataParallel if multiple GPUs are available
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs for DataParallel training.")
    model = nn.DataParallel(model)


model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with distributed training visualization
def train(model, dataloader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)

            # Print GPU and batch info every 100 batches
            if i % 100 == 0:
                print(f"\n--- Batch {i} ---")
                print(f"Input device: {inputs.device}")
                print(f"Model type: {type(model)}")
                print(f"Model device: {next(model.parameters()).device}")
                for gpu in range(torch.cuda.device_count()):
                    print(f"GPU {gpu} memory allocated: {torch.cuda.memory_allocated(gpu) / 1024**2:.2f} MB")
                    print(f"GPU {gpu} memory cached: {torch.cuda.memory_reserved(gpu) / 1024**2:.2f} MB")

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 100 == 99:
                print(f"Epoch {epoch+1}, Batch {i+1}, Loss: {running_loss/100:.3f}")
                running_loss = 0.0

# Train the model
train(model, train_loader, criterion, optimizer, epochs=10)

# Save the model
torch.save(model.module.state_dict(), "resnet50_cifar10_dataparallel.pth")
print("Training complete. Model saved.")

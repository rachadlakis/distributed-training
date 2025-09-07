import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, models
from datasets import load_dataset
from tqdm import tqdm

# Check for available GPUs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = torch.cuda.device_count()
print(f"Using {num_gpus} GPUs for training.")

# Load CIFAR-10 dataset from Hugging Face
dataset = load_dataset("cifar10")
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Use only 10% of the dataset
train_dataset = Subset(train_dataset, range(int(0.1 * len(train_dataset))))
test_dataset = Subset(test_dataset, range(int(0.1 * len(test_dataset))))

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
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]["img"]
        label = self.data[idx]["label"]
        if self.transform:
            img = self.transform(img)
        return img, label

# Create datasets
train_dataset = CIFAR10Dataset(train_dataset, transform=transform)
test_dataset = CIFAR10Dataset(test_dataset, transform=transform)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Load pre-trained ResNet-50 and modify for CIFAR-10
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)  # Replace final layer for 10 classes


# Wrap the model with DataParallel if multiple GPUs are available
if num_gpus > 1:
    print(f"✅ Using {num_gpus} GPUs for DataParallel training.")
    model = nn.DataParallel(model)   # <--- use DebugDataParallel instead of nn.DataParallel
    print(f"Model is wrapped in DataParallel: {isinstance(model, nn.DataParallel)}")
    print(f"Devices available: {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")
else:
    print("⚠️ Using a single GPU or CPU. DataParallel not enabled.")


model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop with progress bar and GPU monitoring
def train(model, dataloader, criterion, optimizer, epochs=5):
    model.train()
    for epoch in range(epochs):
        loop = tqdm(enumerate(dataloader), total=len(dataloader), leave=False)
        running_loss = 0.0
        for i, (inputs, labels) in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            if num_gpus > 1 and i == 0:
                print(f"[Epoch {epoch+1}] Batch size before scatter: {inputs.size(0)}")
                print(f"[Epoch {epoch+1}] Expected per-GPU batch size: {inputs.size(0) // num_gpus}")           

            optimizer.zero_grad()   # Zero out gradients from the previous step
            outputs = model(inputs) # Forward pass: run the current batch through the model
            loss = criterion(outputs, labels) # Compute loss between predictions (outputs) and true labels
            loss.backward()  # Backward pass: compute gradients of the loss w.r.t. model parameters
            optimizer.step() # Update model parameters using the optimizer (based on computed gradients)
            running_loss += loss.item() # Accumulate loss for logging/averaging

            # Update progress bar
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=running_loss/(i+1))

        print(f"Epoch {epoch+1}, Avg Loss: {running_loss/len(dataloader):.3f}")

# Train the model
train(model, train_loader, criterion, optimizer, epochs=5)


# Accessing the Original Model
if isinstance(model, nn.DataParallel):
    original_model = model.module
    print("\nModel was wrapped in DataParallel. Accessing the original model via .module")
    torch.save(original_model.state_dict(), 'model_state.pth')
else:
    original_model = model
    print("\nModel was not wrapped. Saving the model directly.")
    torch.save(original_model.state_dict(), 'model_state.pth')


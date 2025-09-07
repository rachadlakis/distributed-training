# Imports and Initial Setup
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

print(f"PyTorch Version: {torch.__version__}")
print("-" * 30)
# Check for available GPUs
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Found {num_gpus} GPUs.")
    # Set the primary device
    device = torch.device("cuda:0")
else:
    num_gpus = 0
    print("No GPUs found. Running on CPU.")
    device = torch.device("cpu")

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
            print("\tInside the Model: input size", x.size(), "output size", out.size())
        return out

# Data Preparation and Training Loop
input_size = 784
output_size = 10
batch_size = 256  # A larger batch size helps utilize multiple GPUs
learning_rate = 0.01
num_epochs = 20

# Create dummy data
inputs = torch.randn(10000, input_size)
targets = torch.randint(0, output_size, (10000,))
dataset = TensorDataset(inputs, targets)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize and Wrap the Model
model = SimpleModel(input_size, output_size)
if num_gpus > 1:
    print(f"Using {num_gpus} GPUs for training!")
    model = nn.DataParallel(model)
else:
    print("Training on a single device (CPU or 1 GPU).")
model.to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training Loop
print("\nStarting training...")
for epoch in range(num_epochs):
    total_loss = 0
    for i, (batch_inputs, batch_targets) in enumerate(data_loader):
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)
        debug = epoch == 0 and i == 0
        outputs = model(batch_inputs, debug=debug)
        if debug:
            print("Outside: input size", batch_inputs.size(), "output_size", outputs.size())
        loss = criterion(outputs, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')
print("\nTraining finished!")

# Accessing the Original Model
if isinstance(model, nn.DataParallel):
    original_model = model.module
    print("\nModel was wrapped in DataParallel. Accessing the original model via .module")
    torch.save(original_model.state_dict(), 'model_state.pth')
else:
    original_model = model
    print("\nModel was not wrapped. Saving the model directly.")
    torch.save(original_model.state_dict(), 'model_state.pth')

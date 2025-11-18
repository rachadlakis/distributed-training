import argparse                     # For parsing command-line arguments (though unused here)
import os                           # For interacting with the operating system (e.g., env vars, paths)
import sys                          # For system-specific parameters and functions (e.g., exit)
import tempfile                     # For creating temporary files/directories (used on Windows)
from urllib.parse import urlparse   # For parsing URL-style init_method strings

import torch                        # Core PyTorch library
import torch.distributed as dist    # PyTorch distributed communication package
import torch.nn as nn               # Neural network modules
import torch.optim as optim         # Optimization algorithms (e.g., SGD)

from torch.nn.parallel import DistributedDataParallel as DDP  # Wrapper for model parallelism

def verify_min_gpu_count(min_gpus: int = 2) -> bool:
    """ verification that we have at least 2 gpus to run dist examples """
    has_gpu = torch.accelerator.is_available()              # Check if any accelerator (cuda, ROCm, XPU intel, ) is available
    gpu_count = torch.accelerator.device_count()            # Get number of available accelerators
    return has_gpu and gpu_count >= min_gpus                # Return True if enough GPUs exist

class ToyModel(nn.Module):

    def __init__(self):
        super(ToyModel, self).__init__()                    # Initialize parent nn.Module
        self.net1 = nn.Linear(10, 10)                       # First linear layer: 10 → 10
        self.relu = nn.ReLU()                               # ReLU activation function
        self.net2 = nn.Linear(10, 5)                        # Second linear layer: 10 → 5

    def forward(self, x):
        return self.net2(self.relu(self.net1(x)))           # Forward pass: net1 → ReLU → net2

def demo_basic(rank):
    print(
        f"[{os.getpid()}] rank = {dist.get_rank()}, "       # Print process ID and distributed rank
        + f"world_size = {dist.get_world_size()}"           # Print total number of processes
    )

    model = ToyModel().to(rank)                             # Move model to device (GPU/CPU) specified by rank
    ddp_model = DDP(model, device_ids=[rank])               # Wrap model with DDP for distributed training

    loss_fn = nn.MSELoss()                                  # Define mean squared error loss function
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001) # Stochastic Gradient Descent optimizer

    optimizer.zero_grad()                                   # Clear previous gradients
    outputs = ddp_model(torch.randn(20, 10))                # Forward pass with random input batch (20x10)
    labels = torch.randn(20, 5).to(rank)                    # Random target labels (20x5), moved to device
    loss_fn(outputs, labels).backward()                     # Compute loss and backpropagate gradients
    optimizer.step()                                        # Update model parameters

    print(f"training completed in rank {rank}!")            # Confirm training finished for this rank

def main():
    # These are the parameters used to initialize the process group
    env_dict = {
        key: os.environ[key]                                # Extract distributed training env vars
        for key in ("MASTER_ADDR", "MASTER_PORT", "RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE")
    } ## torchrun sets these environment variables for us:
        # MASTER_ADDR (default: 127.0.0.1)
        # MASTER_PORT (default: 29500)
        # RANK (0, 1, 2, ... for each process)
        # LOCAL_RANK (0, 1, ... per machine)
        # WORLD_SIZE (total number of processes)
        # LOCAL_WORLD_SIZE (processes per machine)

    rank = int(env_dict['RANK'])                            # Global rank of this process
    local_rank = int(env_dict['LOCAL_RANK'])                # Local rank (per-node index)
    local_world_size = int(env_dict['LOCAL_WORLD_SIZE'])    # Number of processes on this node

    if sys.platform == "win32":
        # Distributed package only covers collective communications with Gloo
        # backend and FileStore on Windows platform. Set init_method parameter
        # in init_process_group to a local file.
        if "INIT_METHOD" in os.environ.keys():              # Check if user provided init_method
            print(f"init_method is {os.environ['INIT_METHOD']}")  # Log provided init_method
            url_obj = urlparse(os.environ["INIT_METHOD"])   # Parse the URL string
            if url_obj.scheme.lower() != "file":            # Only 'file://' is supported on Windows
                raise ValueError("Windows only supports FileStore")  # Raise error if invalid
            else:
                init_method = os.environ["INIT_METHOD"]     # Use user-provided file path
        else:
            # It is a example application, For convience, we create a file in temp dir.
            temp_dir = tempfile.gettempdir()                # Get system temp directory
            init_method = f"file:///{os.path.join(temp_dir, 'ddp_example')}"  # Auto-generate file path
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}, init_method: {init_method}")  # Log env vars and init_method on Windows
         # Initialize process group (Windows)    
        dist.init_process_group(
            backend="gloo", 
            init_method=init_method, 
            rank=int(env_dict["RANK"]), 
            world_size=int(env_dict["WORLD_SIZE"])
            ) 
         # Initialize process group (Windows)
    else:
        print(f"[{os.getpid()}] Initializing process group with: {env_dict}")  # Log env vars on non-Windows
        acc = torch.accelerator.current_accelerator()       # Get current accelerator (e.g., CUDA)
        backend = torch.distributed.get_default_backend_for_device(acc)  # Choose optimal backend (e.g., nccl for CUDA)
        torch.accelerator.set_device_index(rank)            # Set current device to match rank
        dist.init_process_group(backend=backend)            # Initialize process group (Linux/macOS)

    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "  # Print distributed setup info
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()} \n", end=''
    )

    demo_basic(rank)                                        # Run the distributed training demo

    # Tear down the process group
    dist.destroy_process_group()                            # Clean up distributed resources

if __name__ == "__main__":
    _min_gpu_count = 2                                      # Require at least 2 GPUs
    if not verify_min_gpu_count(min_gpus=_min_gpu_count):   # Check GPU availability
        print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")  # Error if insufficient GPUs
        sys.exit()                                          # Exit program gracefully
    main()                                                  # Run main function if checks pass
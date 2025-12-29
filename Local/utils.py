## check if using GPU
import subprocess

def print_nvidia_smi():
    try:
        result = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
        print(result)
    except FileNotFoundError:
        print("nvidia-smi command not found. Are NVIDIA drivers installed?")
    except subprocess.CalledProcessError as e:
        print(f"Error calling nvidia-smi: {e}")


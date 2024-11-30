import torch  # Import the PyTorch library

# Check if CUDA is available for PyTorch
print("CUDA available:", torch.cuda.is_available())

# Print the name of the GPU device if CUDA is available, otherwise indicate no GPU detected
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")

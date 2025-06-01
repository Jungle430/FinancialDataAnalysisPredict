import torch

def get_device() -> torch.device:
    """
    获取当前可用的设备,优先使用GPU(Nvidia CUDA或Apple Silicon MPS),否则使用CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")  # NVidia GPU
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon GPU
    else:
        return torch.device("cpu")  # 默认使用 CPU
import torch
from diffusers import DiffusionPipeline
from config import BASE_PATH

def setup_environment():
    """
    Checks for GPU availability and sets the device and dtype accordingly.
    Returns:
        device (str): 'cuda' if GPU is available, otherwise 'cpu'.
        dtype (torch.dtype): Appropriate data type for the device.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    return device, dtype

def load_stable_diffusion_pipeline(device, dtype):
    """
    Loads the Stable Diffusion pipeline.
    
    Args:
        device (str): 'cuda' or 'cpu' to load the model on.
        dtype (torch.dtype): Data type to be used for the pipeline.
    
    Returns:
        pipeline (DiffusionPipeline): Loaded Stable Diffusion pipeline.
    """
    pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=dtype)
    pipeline.to(device)
    return pipeline

# stable_diffusion_pipeline.py
from diffusers import DiffusionPipeline

def load_stable_diffusion_pipeline(device, dtype):
    """Load Stable Diffusion pipeline with specified device and dtype."""
    pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=dtype)
    pipeline.to(device)
    return pipeline

if __name__ == "__main__":
    # Import from setup_env
    from setup_env import setup_environment
    device, dtype = setup_environment()
    
    pipeline = load_stable_diffusion_pipeline(device, dtype)
    print("Pipeline loaded successfully")

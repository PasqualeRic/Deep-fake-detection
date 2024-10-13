from diffusers import DiffusionPipeline

def load_stable_diffusion_pipeline(device, dtype):
    """Load Stable Diffusion pipeline with specified device and dtype."""
    pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=dtype)
    pipeline.to(device)
    return pipeline

def generate_image_from_prompt(pipeline, prompt):
    """
    Generates an image from a text prompt using the Stable Diffusion pipeline.
    
    Args:
        pipeline (DiffusionPipeline): Pre-loaded Stable Diffusion pipeline.
        prompt (str): Text prompt to generate an image.
    
    Returns:
        PIL.Image: Generated image.
    """
    generated_image = pipeline(prompt).images[0]  # Generate the image
    return generated_image

if __name__ == "__main__":
    # Example: Load environment and generate an image
    from setup_env import setup_environment
    device, dtype = setup_environment()
    
    pipeline = load_stable_diffusion_pipeline(device, dtype)
    prompt = "A serene mountain landscape at sunrise."
    generated_image = generate_image_from_prompt(pipeline, prompt)
    generated_image.show()  # Display the generated image

from diffusers import DiffusionPipeline
import torch

def load_diffusion_pipeline():
    """
    Carica la pipeline di Stable Diffusion per la generazione di immagini.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=dtype)
    pipeline.to(device)
    return pipeline

def generate_images_with_pipeline(pipeline, prompts, save_paths):
    """
    Genera immagini a partire da una lista di prompt testuali usando la pipeline Stable Diffusion.
    """
    for i, prompt in enumerate(prompts):
        image = pipeline(prompt, num_inference_steps=8).images[0]
        image.save(save_paths[i])

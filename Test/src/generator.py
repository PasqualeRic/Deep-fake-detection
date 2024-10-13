import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Defines the GAN generator model for image generation.
    
    Args:
        nz (int): Latent vector size.
        ngf (int): Number of feature maps in generator.
        nc (int): Number of color channels in the generated image.
    """
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

def load_generator(checkpoint_path, device, nz=100, ngf=64, nc=3):
    """
    Loads the pre-trained GAN generator model from a checkpoint.
    
    Args:
        checkpoint_path (str): Path to the GAN checkpoint.
        device (str): 'cuda' or 'cpu' to load the model on.
        nz (int): Latent vector size.
        ngf (int): Number of feature maps in the generator.
        nc (int): Number of color channels in the generated image.
    
    Returns:
        Generator: The loaded GAN generator model.
    """
    netG = Generator(nz=nz, ngf=ngf, nc=nc).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    netG.load_state_dict(state_dict)
    netG.eval()
    return netG

import torch
import torch.nn as nn
import os
from torchvision.utils import save_image
from path import GAN_CHECKPOINT, SAVE_DIRECTORY_GAN
class Generator(nn.Module):
    """
    Modello generativo per la creazione di immagini tramite GAN.
    Utilizza strati di convoluzione trasposta per generare immagini da un vettore latente.

    Args:
    - ngpu (int): Numero di GPU da utilizzare per il training.
    """
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        """
        Funzione forward per il generatore GAN.

        Args:
        - input (Tensor): Input del vettore latente per la generazione delle immagini.

        Returns:
        - Tensor: Immagini generate.
        """
        return self.main(input)

def generate_images_with_gan():
    """
    Genera immagini utilizzando il modello generatore GAN e salva i risultati in formato PNG.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    generator = Generator(ngpu=1).to(device)
    
    # Carica i pesi e genera immagini
    checkpoint_path = '/path/to/checkpoint_generator.pth'
    generator.load_state_dict(torch.load(GAN_CHECKPOINT, map_location=device))
    generator.eval()
    
    # Genera rumore e immagini
    fixed_noise = torch.randn(2000, 100, 1, 1, device=device)
    with torch.no_grad():
        fake_images = generator(fixed_noise).detach().cpu()
    
    # Salva le immagini generate
    save_directory = '/path/to/images'
    os.makedirs(save_directory, exist_ok=True)
    
    for i in range(fake_images.size(0)):
        save_path = os.path.join(SAVE_DIRECTORY_GAN, f'generated_image_{i+1}.png')
        save_image(fake_images[i], save_path, normalize=True)

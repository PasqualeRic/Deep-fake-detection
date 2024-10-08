#GENERATORE GAN IMMAGINI 
import torch
import os
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# Definisci i parametri utilizzati per l'addestramento
nz = 100  # Dimensione del vettore latente z
ngf = 64  # Feature maps nel generatore
nc = 3    # Numero di canali dell'immagine (3 per immagini RGB)
ngpu = 2  # Numero di GPU
num_images = 2000

# Carica il generatore
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
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

# Inizializza il generatore
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
netG = Generator(ngpu).to(device)

# Carica i pesi del generatore salvato
checkpoint_path_generator = '/kaggle/input/100/pytorch/default/1/checkpoint_generator.pth'
checkpoint = torch.load(checkpoint_path_generator)

# Se i pesi sono stati salvati con DataParallel, rimuovi 'module.' dai nomi
if 'module.' in list(checkpoint.keys())[0]:
    checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

netG.load_state_dict(checkpoint)

# Se hai più GPU, usa DataParallel
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))
    
# Crea la cartella per salvare le immagini
output_folder = '/kaggle/working/generated_images'
os.makedirs(output_folder, exist_ok=True)

# Genera e salva le immagini
with torch.no_grad():
    for i in range(num_images):
        fixed_noise = torch.randn(1, nz, 1, 1, device=device)  # Vettore di rumore
        fake = netG(fixed_noise).detach().cpu()

        # Salva l'immagine
        image_path = os.path.join(output_folder, f'generated_image_{i + 1}.png')
        vutils.save_image(fake, image_path, normalize=True)

print(f"Generated and saved {num_images} images in {output_folder}.")
""" 
Da utilizzare su kaggle se si vogliono salvare le immagini generate
import shutil

# Specifica il percorso della cartella che vuoi comprimere
folder_to_zip = '/kaggle/working/generated_images'  # Sostituisci con il percorso della tua cartella
output_zip = '/kaggle/working/output_folder.zip'

# Crea un file ZIP della cartella
shutil.make_archive(folder_to_zip, 'zip', folder_to_zip) """
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class ImageDataset(Dataset):
    """
    Dataset personalizzato per caricare e preprocessare le immagini da una directory.

    Args:
    - image_dir (str): Percorso alla directory delle immagini.
    - size (int): Dimensione a cui ridimensionare le immagini.
    """
    def __init__(self, image_dir, size):
        self.image_dir = image_dir
        self.size = size
        self.image_names = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_path).convert('RGB')
        image = preprocess_image(image, self.size)
        return image, self.image_names[idx]

def preprocess_image(image, size=64):
    """
    Preprocessa un'immagine ridimensionandola e normalizzandola.

    Args:
    - image (PIL.Image): Immagine da preprocessare.
    - size (int): Dimensione a cui ridimensionare l'immagine.

    Returns:
    - Tensor: Immagine preprocessata come Tensor.
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)

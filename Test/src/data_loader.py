import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    """
    Custom dataset class to load and preprocess images from a directory.
    
    Args:
        img_dir (str): Directory containing the images.
        size (int or None): Size to resize the images, or None to keep original size.
    """
    def __init__(self, img_dir, size=None, compression_quality=100):
        self.img_dir = img_dir
        self.size = size
        self.compression_quality = compression_quality
        self.image_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_names[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.preprocess_image(image)
        return image, self.image_names[idx]

    def preprocess_image(self, image):
        """
        Preprocess the image (resize if size is provided, otherwise keep original size).
        """
        if self.size:
            image = image.resize((self.size, self.size))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image)

def create_dataloader(img_dir, batch_size=32, img_size=None):
    """
    Creates a DataLoader for loading images, optionally resizing them.
    
    Args:
        img_dir (str): Directory containing the images.
        batch_size (int): Batch size for DataLoader.
        img_size (int or None): Size to resize images, or None to keep original size.
    
    Returns:
        DataLoader: DataLoader for the images.
    """
    dataset = ImageDataset(img_dir, size=img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

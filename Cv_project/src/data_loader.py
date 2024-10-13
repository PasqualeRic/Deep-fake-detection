import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    """
    Custom dataset class to load and preprocess images from a directory.
    
    Args:
        img_dir (str): Directory containing the images.
        size (int): Size to resize the images.
        compression_quality (int): Quality of JPEG compression (100 means no compression).
    """
    def __init__(self, img_dir, size, compression_quality=100):
        self.img_dir = img_dir
        self.size = size
        self.compression_quality = compression_quality
        self.image_names = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.image_names[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.preprocess_image(image, self.size, self.compression_quality)
        return image, self.image_names[idx]

    def preprocess_image(self, image, size, compression_quality):
        """
        Preprocesses and compresses the image.
        
        Args:
            image (PIL.Image): Input image to preprocess.
            size (int): Desired size for resizing.
            compression_quality (int): Quality of compression (100 = no compression).
        
        Returns:
            Tensor: Preprocessed image as a Tensor.
        """
        transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Compress and resize the image
        temp_img_path = '/tmp/temp_image.jpg'
        image = image.resize((size, size))
        image.save(temp_img_path, format='JPEG', quality=compression_quality)
        compressed_image = Image.open(temp_img_path).convert('RGB')
        return transform(compressed_image)

def create_dataloader_with_compression(img_dir, batch_size=32, img_size=64, compression_quality=100):
    """
    Creates a DataLoader for loading images with compression applied.
    
    Args:
        img_dir (str): Directory containing the images.
        batch_size (int): Batch size for DataLoader.
        img_size (int): Size to resize images.
        compression_quality (int): Quality of JPEG compression for preprocessing.
    
    Returns:
        DataLoader: DataLoader for the images.
    """
    dataset = ImageDataset(img_dir, img_size, compression_quality=compression_quality)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

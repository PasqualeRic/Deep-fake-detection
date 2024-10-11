from PIL import Image
from torchvision import transforms

def preprocess_and_compress_image(image, size=64, compression_quality=100):
    """
    Preprocessa e comprime un'immagine ridimensionandola e applicando una compressione JPEG.

    Args:
    - image (PIL.Image): Immagine da preprocessare.
    - size (int): Dimensione dell'immagine (default: 64).
    - compression_quality (int): Qualità JPEG (100 = nessuna compressione).

    Returns:
    - Tensor: Immagine preprocessata.
    """
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    temp_img_path = '/path/to/temp_image.jpg'
    image = image.resize((size, size))
    image.save(temp_img_path, format='JPEG', quality=compression_quality)
    
    compressed_image = Image.open(temp_img_path).convert('RGB')
    return transform(compressed_image)

import torch
from torchvision import models
from sklearn.externals import joblib

def load_vgg16_model(device):
    """
    Loads a pre-trained VGG16 model for feature extraction.
    
    Args:
        device (str): 'cuda' or 'cpu' to load the model on.
    
    Returns:
        torch.nn.Module: VGG16 model with the final classification layer removed.
    """
    vgg16 = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).to(device)
    vgg16.classifier = vgg16.classifier[:-1]  # Remove the final classification layer
    return vgg16.eval()

def extract_features(images, model, device):
    """
    Extracts features from a batch of images using the VGG16 model.
    
    Args:
        images (Tensor): Batch of images to extract features from.
        model (torch.nn.Module): Pre-trained VGG16 model.
        device (str): 'cuda' or 'cpu' for inference.
    
    Returns:
        np.ndarray: Extracted features as a numpy array.
    """
    with torch.no_grad():
        images = images.to(device)
        features = model(images)
    return features.cpu().numpy()

def load_svm_classifier(path):
    """
    Loads a pre-trained SVM classifier.
    
    Args:
        path (str): Path to the SVM checkpoint file.
    
    Returns:
        sklearn.svm.SVC: Loaded SVM classifier.
    """
    return joblib.load(path)

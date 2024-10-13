import torch
import numpy as np
from torchvision import models
import matplotlib.pyplot as plt
from torchmetrics.image.fid import FrechetInceptionDistance
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_inception_score(images, splits=10, device='cuda'):
    """
    Calculates the Inception Score for generated images.
    
    Args:
        images (Tensor): Batch of generated images.
        splits (int): Number of splits for IS calculation.
        device (str): 'cuda' or 'cpu' for inference.
    
    Returns:
        tuple: Mean and standard deviation of the Inception Score.
    """
    inception_model = models.inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()

    def get_pred(x):
        x = inception_model(x)
        return torch.nn.functional.softmax(x, dim=1).data.cpu().numpy()

    N = len(images)
    preds = np.zeros((N, 1000))
    for i in range(0, N, 32):
        batch_images = images[i:i+32].to(device)
        preds[i:i+32] = get_pred(batch_images)

    scores = []
    for i in range(splits):
        part = preds[i * (N // splits): (i + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        kl_div = part * (np.log(part) - np.log(py))
        kl_div = np.sum(kl_div, axis=1)
        scores.append(np.exp(np.mean(kl_div)))

    return np.mean(scores), np.std(scores)

def calculate_fid(real_images_loader, generated_images_loader, device='cuda'):
    """
    Calculates the Fréchet Inception Distance (FID) between real and generated images.
    
    Args:
        real_images_loader (DataLoader): DataLoader for real images.
        generated_images_loader (DataLoader): DataLoader for generated images.
        device (str): 'cuda' or 'cpu' for inference.
    
    Returns:
        float: FID score.
    """
    fid = FrechetInceptionDistance(feature=64).to(device)

    for real_images in real_images_loader:
        fid.update(real_images.to(device), real=True)

    for generated_images in generated_images_loader:
        fid.update(generated_images.to(device), real=False)

    return fid.compute().item()

def save_fis_results(fis_score, fis_txt_file, fis_plot_file):
    """
    Saves the Feature Importance Score (FIS) to a file and generates a plot.
    
    Args:
        fis_score (float): FIS value to be saved.
        fis_txt_file (str): Path to save the FIS score as a text file.
        fis_plot_file (str): Path to save the FIS plot.
    """
    with open(fis_txt_file, 'w') as f:
        f.write(f"Feature Importance Score (FIS): {fis_score:.4f}\n")

    plt.figure(figsize=(6, 4))
    plt.plot([1], [fis_score], marker='o', linestyle='-', color='b')
    plt.title("Feature Importance Score (FIS)")
    plt.savefig(fis_plot_file)
    plt.close()

def calculate_classification_metrics(y_true, y_pred, save_path):
    """
    Calculates classification metrics (Accuracy, Precision, Recall, F1 score) and saves them to a file.
    
    Args:
        y_true (list): List of true labels (real = 1, generated = 0).
        y_pred (list): List of predicted labels (from the classifier).
        save_path (str): Path to save the results.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    # Save the results to a text file
    with open(save_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

    print(f"Classification metrics saved to {save_path}")

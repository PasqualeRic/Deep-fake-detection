import torch

def extract_features(images, model):
    """
    Estrae le caratteristiche (features) dalle immagini utilizzando un modello pre-addestrato.

    Args:
    - images (Tensor): Batch di immagini preprocessate.
    - model (torch.nn.Module): Modello pre-addestrato per l'estrazione delle caratteristiche.

    Returns:
    - np.ndarray: Caratteristiche estratte dalle immagini.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device)
    with torch.no_grad():
        features = model(images)
    return features.flatten(1).cpu().numpy()

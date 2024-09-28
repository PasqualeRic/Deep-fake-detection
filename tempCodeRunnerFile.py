import torch
from diffusers import DiffusionPipeline
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from pycocotools.coco import COCO
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Percorso al tuo dataset COCO
coco_root = '/Users/pasqualericciulli/Downloads/train2017'  # Cambia questo con il tuo percorso
coco_annotation_file = '/Users/pasqualericciulli/Downloads/annotations/instances_train2017.json'
coco = COCO(coco_annotation_file)

# Verifica se c'è una GPU disponibile
device = "cuda" if torch.cuda.is_available() else "cpu"

# Usa torch.float16 solo se c'è una GPU, altrimenti torch.float32
dtype = torch.float16 if device == "cuda" else torch.float32

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=dtype)

# Se possibile, utilizza 'accelerate' per migliorare le prestazioni
try:
    pipeline.enable_attention_slicing()
    print("Accelerate abilitato per migliorare le prestazioni.")
except ImportError:
    print("Attenzione: 'accelerate' non trovato. Per migliorare le prestazioni, installalo con 'pip install accelerate'.")

pipeline.to(device)

# Funzione per generare un'immagine
def generate_image(prompt):
    image = pipeline(prompt, num_inference_steps=8).images[0]
    return image.convert("RGB")

# Funzione per applicare un filtro passa-alto
def high_pass_filter(image):
    image_np = np.array(image.convert('L'))  # Converte in scala di grigi
    f_transform = np.fft.fft2(image_np)
    f_transform_shifted = np.fft.fftshift(f_transform)
    
    # Definisci una maschera per il filtro passa-alto
    rows, cols = image_np.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 30  # Raggio del filtro passa-alto
    mask[crow-r:crow+r, ccol-r:ccol+r] = 0
    
    # Applica la maschera
    f_transform_shifted_filtered = f_transform_shifted * mask
    f_ishift = np.fft.ifftshift(f_transform_shifted_filtered)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    
    return img_back

# Funzione per calcolare il fingerprint spettrale
def get_spectral_fingerprint(image):
    filtered_image = high_pass_filter(image)
    f_transform = np.fft.fft2(filtered_image)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_transform_shifted)
    magnitude_spectrum = np.log(magnitude_spectrum + 1)
    return magnitude_spectrum

# Funzione per estrarre caratteristiche da VGG16
def extract_vgg_features(image, model):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0)  # Aggiunge la dimensione batch
    with torch.no_grad():
        features = model(input_tensor)
    return features.flatten().numpy()  # Appiattisce le caratteristiche

# Carica il modello VGG16 pre-addestrato
vgg16 = models.vgg16(pretrained=True)
vgg16.classifier = vgg16.classifier[:-1]  # Rimuove l'ultimo strato
vgg16.eval()

# Estrazione delle caratteristiche per l'addestramento del classificatore
features_list = []
labels = []

# Esempio di utilizzo: prendi alcune immagini dal dataset
image_ids = coco.getImgIds()
sample_ids = image_ids[:10]  # Prendi 10 immagini come esempio

for img_id in sample_ids:
    img_data = coco.imgs[img_id]
    img_path = os.path.join(coco_root, img_data['file_name'])

    # Carica l'immagine reale
    image_real = Image.open(img_path).convert("RGB")

    # Estrai le caratteristiche dall'immagine reale
    real_features = extract_vgg_features(image_real, vgg16)
    features_list.append(real_features)
    labels.append(1)  # 1 per immagini reali

    # Ottieni le categorie associate all'immagine per generare un'immagine fake
    ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
    anns = coco.loadAnns(ann_ids)
    categories = [coco.cats[ann['category_id']]['name'] for ann in anns]
    prompt = ", ".join(categories)

    # Genera un'immagine fake usando il prompt
    image_fake = generate_image(prompt)

    # Estrai le caratteristiche dall'immagine fake
    fake_features = extract_vgg_features(image_fake, vgg16)
    features_list.append(fake_features)
    labels.append(0)  # 0 per immagini fake

# Dividi i dati in training e test set
X_train, X_test, y_train, y_test = train_test_split(features_list, labels, test_size=0.2, random_state=42)

# Addestra un classificatore SVM
classifier = SVC(kernel='linear')
classifier.fit(X_train, y_train)

# Fai previsioni sul test set
y_pred = classifier.predict(X_test)

# Mostra il report di classificazione
print(classification_report(y_test, y_pred))

# Ora integriamo la parte di riconoscimento
# Esempio di utilizzo per riconoscere se un'immagine è reale o fake
for img_id in sample_ids:
    img_data = coco.imgs[img_id]
    img_path = os.path.join(coco_root, img_data['file_name'])

    # Carica l'immagine reale
    image_real = Image.open(img_path).convert("RGB")

    # Estrai le caratteristiche dall'immagine reale
    real_features = extract_vgg_features(image_real, vgg16)

    # Fai la previsione
    prediction = classifier.predict(real_features.reshape(1, -1))  # Reshape per SVM
    result = "Real" if prediction[0] == 1 else "Fake"

    print(f"Risultato per l'immagine ID {img_id}: {result}")

    # Genera un'immagine fake usando il prompt
    prompt = " ".join(categories)
    image_fake = generate_image(prompt)

    # Estrai le caratteristiche dall'immagine fake
    fake_features = extract_vgg_features(image_fake, vgg16)

    # Fai la previsione
    prediction_fake = classifier.predict(fake_features.reshape(1, -1))
    result_fake = "Real" if prediction_fake[0] == 1 else "Fake"

    print(f"Risultato per l'immagine generata: {result_fake}")

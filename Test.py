import torch
from diffusers import DiffusionPipeline
from torchvision import models, transforms
from PIL import Image
import os
from pycocotools.coco import COCO
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import json  # Importa il modulo json per salvare i dati

# Percorso al dataset COCO
coco_root = '/content/drive/MyDrive/dataset_ridotto_test/test2017' 
coco_annotation_file = '/content/drive/MyDrive/dataset_ridotto_test/instances_test2017_subset.json'
coco = COCO(coco_annotation_file)

# Verifica della GPU disponibile
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=dtype)
pipeline.to(device)

# Caricamento del modello addestrato
classifier = joblib.load('/content/drive/MyDrive/svm_classifier.pkl')

# Funzione per generare immagini a partire da un prompt
def generate_image(prompt):
    image = pipeline(prompt, num_inference_steps=8).images[0]
    return image.convert("RGB")

# Funzione per estrarre le caratteristiche con VGG16
def extract_vgg_features(image, model):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = model(input_tensor)
    return features.flatten().cpu().numpy()

# Funzione per applicare compressione
def apply_compression(image, quality):
    img_byte_arr = io.BytesIO()  # Usa un buffer in memoria
    image.save(img_byte_arr, format='JPEG', quality=quality)
    img_byte_arr.seek(0)  # Torna all'inizio del buffer
    return Image.open(img_byte_arr)  # Restituisci l'immagine

# Funzione per aggiungere rumore gaussiano
def add_gaussian_noise(image, mean=0, std=25):
    image_np = np.array(image)
    noise = np.random.normal(mean, std, image_np.shape).astype(np.uint8)
    noisy_image = np.clip(image_np + noise, 0, 255)
    return Image.fromarray(noisy_image)

# Funzione per visualizzare le immagini
def plot_images(image_real, image_fake, image_real_noisy, image_fake_noisy, image_real_compressed_80, image_fake_compressed_80, image_real_compressed_60, image_fake_compressed_60):
    plt.figure(figsize=(15, 12))

    # Immagine reale originale
    plt.subplot(4, 2, 1)
    plt.imshow(image_real)
    plt.title("Real Image")
    plt.axis('off')

    # Immagine fake originale
    plt.subplot(4, 2, 2)
    plt.imshow(image_fake)
    plt.title("Fake Image")
    plt.axis('off')

    # Immagine reale con rumore
    plt.subplot(4, 2, 3)
    plt.imshow(image_real_noisy)
    plt.title("Real Image with Noise")
    plt.axis('off')

    # Immagine fake con rumore
    plt.subplot(4, 2, 4)
    plt.imshow(image_fake_noisy)
    plt.title("Fake Image with Noise")
    plt.axis('off')

    # Immagine reale compressa 80
    plt.subplot(4, 2, 5)
    plt.imshow(image_real_compressed_80)
    plt.title("Compressed Real (80%)")
    plt.axis('off')

    # Immagine fake compressa 80
    plt.subplot(4, 2, 6)
    plt.imshow(image_fake_compressed_80)
    plt.title("Compressed Fake (80%)")
    plt.axis('off')

    # Immagine reale compressa 60
    plt.subplot(4, 2, 7)
    plt.imshow(image_real_compressed_60)
    plt.title("Compressed Real (60%)")
    plt.axis('off')

    # Immagine fake compressa 60
    plt.subplot(4, 2, 8)
    plt.imshow(image_fake_compressed_60)
    plt.title("Compressed Fake (60%)")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Caricamento di VGG16 pre-addestrato e rimozione dell'ultimo layer
vgg16 = models.vgg16(pretrained=True)
vgg16.classifier = vgg16.classifier[:-1]  # Rimuovi l'ultimo strato
vgg16.eval().to(device)

# Variabili per classificazione e metriche
y_true = []
y_pred = []
y_pred_compression_80 = []
y_pred_compression_60 = []
y_pred_noisy = []

# Dizionario per memorizzare le metriche
metrics = {
    'original': {},
    'compressed_80': {},
    'compressed_60': {},
    'noisy': {}
}

# Iterazione sulle immagini nel dataset COCO
image_ids = coco.getImgIds()
sample_ids = image_ids[:50]  # Campiona solo 500 immagini
for i, img_id in enumerate(sample_ids, start=1):
    img_data = coco.imgs[img_id]
    img_path = os.path.join(coco_root, img_data['file_name'])

    try:
        # Carica l'immagine reale ed estrai le caratteristiche
        image_real = Image.open(img_path).convert("RGB")
        real_features = extract_vgg_features(image_real, vgg16).reshape(1, -1)  # Aggiungi dimensione batch

        # Estrai il prompt dalle annotazioni
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        categories = [coco.cats[ann['category_id']]['name'] for ann in anns]
        prompt = ", ".join(categories)

        # Genera l'immagine fake e ottieni le caratteristiche
        image_fake = generate_image(prompt)
        fake_features = extract_vgg_features(image_fake, vgg16).reshape(1, -1)

        # Aggiungi rumore gaussiano
        image_real_noisy = add_gaussian_noise(image_real)
        image_fake_noisy = add_gaussian_noise(image_fake)

        # Applica compressione
        image_real_compressed_80 = apply_compression(image_real, quality=80)
        image_fake_compressed_80 = apply_compression(image_fake, quality=80)

        image_real_compressed_60 = apply_compression(image_real, quality=60)
        image_fake_compressed_60 = apply_compression(image_fake, quality=60)

        # Estrai le caratteristiche dalle immagini compresse
        real_features_compression_80 = extract_vgg_features(image_real_compressed_80, vgg16).reshape(1, -1)
        fake_features_compression_80 = extract_vgg_features(image_fake_compressed_80, vgg16).reshape(1, -1)

        real_features_compression_60 = extract_vgg_features(image_real_compressed_60, vgg16).reshape(1, -1)
        fake_features_compression_60 = extract_vgg_features(image_fake_compressed_60, vgg16).reshape(1, -1)

        # Estrai le caratteristiche dalle immagini rumorose
        real_features_noisy = extract_vgg_features(image_real_noisy, vgg16).reshape(1, -1)
        fake_features_noisy = extract_vgg_features(image_fake_noisy, vgg16).reshape(1, -1)

        #---------- classificazione ------------

        # Classifica separatamente le immagini senza compressione
        real_prediction = classifier.predict(real_features)[0]
        fake_prediction = classifier.predict(fake_features)[0]

        # Classifica le immagini rumorose
        real_prediction_noisy = classifier.predict(real_features_noisy)[0]
        fake_prediction_noisy = classifier.predict(fake_features_noisy)[0]

        # Classifica le immagini compresse
        real_prediction_compression_80 = classifier.predict(real_features_compression_80)[0]
        fake_prediction_compression_80 = classifier.predict(fake_features_compression_80)[0]

        real_prediction_compression_60 = classifier.predict(real_features_compression_60)[0]
        fake_prediction_compression_60 = classifier.predict(fake_features_compression_60)[0]

        # Aggiorna le liste per le metriche
        y_true.extend([1, 0])  # 1 per reale, 0 per fake
        y_pred.extend([real_prediction, fake_prediction])
        y_pred_compression_80.extend([real_prediction_compression_80, fake_prediction_compression_80])
        y_pred_compression_60.extend([real_prediction_compression_60, fake_prediction_compression_60])
        y_pred_noisy.extend([real_prediction_noisy, fake_prediction_noisy])

        # Calcola le metriche e memorizzale
        metrics['original']['precision'] = precision_score(y_true, y_pred)
        metrics['original']['recall'] = recall_score(y_true, y_pred)
        metrics['original']['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['original']['f1'] = f1_score(y_true, y_pred)

        metrics['compressed_80']['precision'] = precision_score(y_true, y_pred_compression_80)
        metrics['compressed_80']['recall'] = recall_score(y_true, y_pred_compression_80)
        metrics['compressed_80']['accuracy'] = accuracy_score(y_true, y_pred_compression_80)
        metrics['compressed_80']['f1'] = f1_score(y_true, y_pred_compression_80)

        metrics['compressed_60']['precision'] = precision_score(y_true, y_pred_compression_60)
        metrics['compressed_60']['recall'] = recall_score(y_true, y_pred_compression_60)
        metrics['compressed_60']['accuracy'] = accuracy_score(y_true, y_pred_compression_60)
        metrics['compressed_60']['f1'] = f1_score(y_true, y_pred_compression_60)

        metrics['noisy']['precision'] = precision_score(y_true, y_pred_noisy)
        metrics['noisy']['recall'] = recall_score(y_true, y_pred_noisy)
        metrics['noisy']['accuracy'] = accuracy_score(y_true, y_pred_noisy)
        metrics['noisy']['f1'] = f1_score(y_true, y_pred_noisy)
        # Determina la classificazione come testo
        real_label = 'Real' if real_prediction == 1 else 'Fake'
        fake_label = 'Real' if fake_prediction == 1 else 'Fake'

        # Determina la classificazione come testo per le immagini rumorose
        real_label_noisy = 'Real' if real_prediction_noisy == 1 else 'Fake'
        fake_label_noisy = 'Real' if fake_prediction_noisy == 1 else 'Fake'

        # Classifica le immagini compresse
        real_prediction_compression_80 = classifier.predict(real_features_compression_80)[0]
        fake_prediction_compression_80 = classifier.predict(fake_features_compression_80)[0]

        # Determina la classificazione come testo per le immagini compresse a 80%
        real_label_compression_80 = 'Real' if real_prediction_compression_80 == 1 else 'Fake'
        fake_label_compression_80 = 'Real' if fake_prediction_compression_80 == 1 else 'Fake'

        real_prediction_compression_60 = classifier.predict(real_features_compression_60)[0]
        fake_prediction_compression_60 = classifier.predict(fake_features_compression_60)[0]

        # Determina la classificazione come testo per le immagini compresse a 60%
        real_label_compression_60 = 'Real' if real_prediction_compression_60 == 1 else 'Fake'
        fake_label_compression_60 = 'Real' if fake_prediction_compression_60 == 1 else 'Fake'

        # Stampa i risultati
        print(f"Immagine ID {img_id}: Reale -> {real_label}, Fake -> {fake_label}")
        print(f"Immagine compressa 80 ID {img_id}: Reale -> {real_label_compression_80}, Fake -> {fake_label_compression_80}")
        print(f"Immagine compressa 60 ID {img_id}: Reale -> {real_label_compression_60}, Fake -> {fake_label_compression_60}")
        print(f"Immagine Gaussian noisy ID {img_id}: Reale -> {real_label_noisy}, Fake -> {fake_label_noisy}")
        print(i)

        # Visualizza le immagini
        plot_images(image_real, image_fake, image_real_noisy, image_fake_noisy, image_real_compressed_80, image_fake_compressed_80,image_real_compressed_60, image_fake_compressed_60)

    except Exception as e:
        print(f"Errore nell'elaborazione dell'immagine ID {img_id}: {e}")

# Salva le metriche in un file JSON
with open('/content/drive/MyDrive/metrics.json', 'w') as f:
    json.dump(metrics, f)

# Stampa le metriche
print(f"\nMetriche del modello (senza compressione):")
print(f"Precisione: {metrics['original']['precision']:.4f}")
print(f"Richiamo: {metrics['original']['recall']:.4f}")
print(f"Accuratezza: {metrics['original']['accuracy']:.4f}")
print(f"F1 Score: {metrics['original']['f1']:.4f}")

print(f"\nMetriche del modello (con compressione 80%):")
print(f"Precisione: {metrics['compressed_80']['precision']:.4f}")
print(f"Richiamo: {metrics['compressed_80']['recall']:.4f}")
print(f"Accuratezza: {metrics['compressed_80']['accuracy']:.4f}")
print(f"F1 Score: {metrics['compressed_80']['f1']:.4f}")

print(f"\nMetriche del modello (con compressione 60%):")
print(f"Precisione: {metrics['compressed_60']['precision']:.4f}")
print(f"Richiamo: {metrics['compressed_60']['recall']:.4f}")
print(f"Accuratezza: {metrics['compressed_60']['accuracy']:.4f}")
print(f"F1 Score: {metrics['compressed_60']['f1']:.4f}")

print(f"\nMetriche del modello (con rumore):")
print(f"Precisione: {metrics['noisy']['precision']:.4f}")
print(f"Richiamo: {metrics['noisy']['recall']:.4f}")
print(f"Accuratezza: {metrics['noisy']['accuracy']:.4f}")
print(f"F1 Score: {metrics['noisy']['f1']:.4f}")

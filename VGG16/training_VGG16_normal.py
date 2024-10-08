#pip install diffusers
#pip install pycocotools
#In this file I generete the images with stable diffusion using the coco's dataset and the file training_VGG16_64_256 train the classifier with upsampling and downsampling images
#and to avoid to generate others 2000 images with stable I save this images and utilise them in that file
import torch
from diffusers import DiffusionPipeline
from torchvision import models, transforms
from PIL import Image
import os
from pycocotools.coco import COCO
import numpy as np
from sklearn.svm import SVC
import joblib
import concurrent.futures
from accelerate import Accelerator

# The train2017 and instances_train2017_subset are subsets of COCO's dataset, they contain only 2000 images
coco_root = '/kaggle/input/coco-2017-dataset/coco2017/train2017'
coco_annotation_file = '/kaggle/input/coco-2017-dataset/coco2017/annotations/instances_train2017.json'
coco = COCO(coco_annotation_file)
gan_root = '/kaggle/input/gan-2000-images-64x64' 

# Check if I have a GPU available 
device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = torch.float16 if device == "cuda" else torch.float32

# Inizializza Accelerator per la gestione delle GPU
accelerator = Accelerator()

# Carica il modello di diffusione
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=dtype)
pipeline.to(accelerator.device)

# Funzione per generare immagini
def generate_image(prompt):
    image = pipeline(prompt, num_inference_steps=8).images[0]
    return image.convert("RGB")

# Funzione per estrarre caratteristiche con VGG
def extract_vgg_features(image, model):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(accelerator.device)  # add the dimension batch and transfer to the GPU
    with torch.no_grad():
        features = model(input_tensor)
    return features.flatten().cpu().numpy()

# Funzione per salvare le immagini fake
def save_image_fake(image, save_dir, img_id):
    os.makedirs(save_dir, exist_ok=True)  # Crea la cartella se non esiste
    save_path = os.path.join(save_dir, f"fake_image_{img_id}.jpg")
    image.save(save_path, format="JPEG")
    print(f"Immagine salvata in: {save_path}")
# Funzione per salvare le immagini reali
def save_image_real(image, save_dir, img_id):
    os.makedirs(save_dir, exist_ok=True)  # Crea la cartella se non esiste
    save_path = os.path.join(save_dir, f"real_image_{img_id}.jpg")
    image.save(save_path, format="JPEG")
    print(f"Immagine salvata in: {save_path}")


# Load a VGG16 pre-trained model and remove the last layer
vgg16 = models.vgg16(pretrained=True)
vgg16.classifier = vgg16.classifier[:-1]  # remove the last layer
vgg16.eval().to(accelerator.device)  # put in eval modality and transfer to the GPU

# Directory di salvataggio
real_dir = "/kaggle/working/real_images"
fake_dir = "/kaggle/working/fake_images"
classifier = SVC(kernel='linear')  # Usa tutte le CPU disponibili
features_list = []
labels = []
last_step = 0

# Estrai immagini da COCO
image_ids = coco.getImgIds()
sample_ids = image_ids[last_step:]

step = 0

# Generazione immagini e raccolta caratteristiche tqdm
for img_id in sample_ids:
    img_data = coco.imgs[img_id]
    img_path = os.path.join(coco_root, img_data['file_name'])
    if step >= 2000:  # Limita a 2000 immagini
        break

    try:
        # Immagine reale
        image_real = Image.open(img_path).convert("RGB")
        real_features = extract_vgg_features(image_real, vgg16)
        features_list.append(real_features)
        labels.append(1)  # etichetta per le immagini reali
        save_image_real(image_real, real_dir, img_id)

        # Ottieni categorie per generare immagini fake
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        categories = [coco.cats[ann['category_id']]['name'] for ann in anns]
        prompt = ", ".join(categories)

        # Generazione immagini in parallelo
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_image = executor.submit(generate_image, prompt)
            image_fake = future_image.result()  # Aspetta il risultato
            save_image_fake(image_fake, fake_dir, img_id)

        fake_features = extract_vgg_features(image_fake, vgg16)
        features_list.append(fake_features)
        labels.append(0)  # etichetta per le immagini fake

        step += 1

    except Exception as e:
        print(f"Errore nell'elaborazione dell'immagine ID {img_id}: {e}")

step = 0
# Elaborazione delle immagini GAN
gan_images = os.listdir(gan_root)
for gan_img_name in gan_images:
    img_path = os.path.join(gan_root, gan_img_name)
    try:
        # Immagine GAN
        gan_image = Image.open(img_path).convert("RGB")
        gan_features = extract_vgg_features(gan_image, vgg16)
        features_list.append(gan_features)
        labels.append(0)  # etichetta per le immagini GAN
        print(step)
        step += 1

    except Exception as e:
        print(f"Errore nell'elaborazione dell'immagine ID {img_id}: {e}")

# Addestra il classificatore SVM
classifier.fit(features_list, labels)

# Salva il modello finale
joblib.dump(classifier, 'svm_classifier.pkl')
print("Modello SVM salvato come 'svm_classifier.pkl'")

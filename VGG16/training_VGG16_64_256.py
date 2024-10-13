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

# Train dataset
coco_root = '/kaggle/input/coco-subset'
gan_root = '/kaggle/input/gan-2000-images-64x64' 
stable_root = '/kaggle/input/stable-diffusion'
# Check if I have a GPU available 
device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = torch.float16 if device == "cuda" else torch.float32

#Initialise Accelerator for GPU management
accelerator = Accelerator()

# Function to save images
def save_image_256(image, save_dir, img_id):
    os.makedirs(save_dir, exist_ok=True)  
    save_path = os.path.join(save_dir, f"image_256x256{img_id}.jpg")
    image.save(save_path, format="JPEG")
    print(f"Immagine salvata in: {save_path}")
def save_image_64(image, save_dir, img_id):
    os.makedirs(save_dir, exist_ok=True) 
    save_path = os.path.join(save_dir, f"image_64x64{img_id}.jpg")
    image.save(save_path, format="JPEG")
    print(f"Immagine salvata in: {save_path}")
def save_image_fake(image, save_dir, img_id):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"fake_image_{img_id}.jpg")
    image.save(save_path, format="JPEG")
    print(f"Immagine salvata in: {save_path}")
def save_image_real(image, save_dir, img_id):
    os.makedirs(save_dir, exist_ok=True) 
    save_path = os.path.join(save_dir, f"real_image_{img_id}.jpg")
    image.save(save_path, format="JPEG")
    print(f"Immagine salvata in: {save_path}")

#Function for upsampling to 256x256 if the size is smaller than 256x256
def upsample_to_256(image):
    if image.size[0] < 256 or image.size[1] < 256:
        return image.resize((256, 256), Image.BICUBIC)
    return image

#Function for downsampling to 256x256 if the size is larger than 256x256
def downsample_to_256(image):
    if image.size[0] > 256 or image.size[1] > 256:
        return image.resize((256, 256), Image.LANCZOS)
    return image
#Function for downsampling to 64x64 if the size is larger than 64x64
def downsample_to_64(image):
    if image.size[0] > 64 or image.size[1] > 64:
        return image.resize((64, 64), Image.LANCZOS)
    return image

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
# Load a VGG16 pre-trained model and remove the last layer
vgg16 = models.vgg16(pretrained=True)
vgg16.classifier = vgg16.classifier[:-1]  # remove the last layer
vgg16.eval().to(accelerator.device)  # put in eval modality and transfer to the GPU

#directory to saving images
original_dir = "/kaggle/working/original"
gan_dir = "/kaggle/working/gan"
stable_dir = "/kaggle/working/stable"

classifier_256 = SVC(kernel='linear')
classifier_64 = SVC(kernel='linear')

features_list_256 = []
features_list_64 = []
labels_256 = []
labels_64 = []

# Extract images from coco
coco_images = os.listdir(coco_root)

# Extract images from gan
gan_images = os.listdir(gan_root)

# Extract images from stable
stable_images = os.listdir(stable_root)

step = 0
original = True
gan = True
stable = True
#We have two classifier, one for 256x256 and one for 64x64
#Append coco's image
print("sei in coco")
for coco_img_name in coco_images:
    img_path = os.path.join(coco_root, coco_img_name)
    try:
        # real image
        img = Image.open(img_path).convert("RGB")
        image_real_256 = downsample_to_256(img)
        image_real_64 = downsample_to_64(img)
        #This if is used just to save one image for all the dimension
        if(original == True):
            save_image_real(img, original_dir, coco_img_name)
            save_image_256(image_real_256, original_dir, coco_img_name)
            save_image_64(image_real_64, original_dir, coco_img_name)
            original = False
        #Feature extraction 
        real_features_256 = extract_vgg_features(image_real_256, vgg16)
        real_features_64 = extract_vgg_features(image_real_64, vgg16)
        
        features_list_256.append(real_features_256)
        features_list_64.append(real_features_64)
        
        labels_256.append(1)  #we append 1 for the real images
        labels_64.append(1)
        step += 1

    except Exception as e:
        print(f"Errore nell'elaborazione dell'immagine ID {coco_img_name}: {e}")
        
#Append stable's image
step = 0
print("sei in stable")
for stable_img_name in stable_images:
    img_path = os.path.join(stable_root, stable_img_name)

    try:
        # Stable diffusion images
        img = Image.open(img_path).convert("RGB")
        image_stable_256 = downsample_to_256(img)
        image_stable_64 = downsample_to_64(img)
        if(stable == True):
            save_image_fake(img, stable_dir, stable_img_name)
            save_image_256(image_stable_256, stable_dir, stable_img_name)
            save_image_64(image_stable_64, stable_dir, stable_img_name)
            stable = False
            
        stable_features_256 = extract_vgg_features(image_stable_256, vgg16)
        stable_features_64 = extract_vgg_features(image_stable_64, vgg16)
        
        features_list_256.append(stable_features_256)
        features_list_64.append(stable_features_64)
        
        labels_256.append(0)  #0 for generated
        labels_64.append(0)
        step += 1
    except Exception as e:
        print(f"Errore nell'elaborazione dell'immagine Stable {stable_img_name}: {e}")


# Append gan's image
step = 0
print("sei in gan")
for gan_img_name in gan_images:
    img_path = os.path.join(gan_root, gan_img_name)

    try:
        # gan images
        img = Image.open(img_path).convert("RGB")
        image_gan_256 = upsample_to_256(img)
        if(gan == True):
            save_image_fake(img, gan_dir, gan_img_name)
            save_image_256(image_gan_256, gan_dir, gan_img_name)
            save_image_64(img, gan_dir, gan_img_name)
            gan = False
        gan_features_256 = extract_vgg_features(image_gan_256, vgg16)
        gan_features_64 = extract_vgg_features(img, vgg16)
        
        features_list_256.append(gan_features_256)
        features_list_64.append(gan_features_64)
        
        labels_256.append(0)  # 0 for generated
        labels_64.append(0)
        step += 1
    except Exception as e:
        print(f"Errore nell'elaborazione dell'immagine Gan {gan_img_name}: {e}")


# train the SVM classifiers, one with only 256x256 images and the other one with 64x64
classifier_256.fit(features_list_256, labels_256)
classifier_64.fit(features_list_256, labels_64)

# Store the final models
joblib.dump(classifier_256, 'svm_classifier_256.pkl')
print("Modello SVM salvato come 'svm_classifier_256.pkl'")
joblib.dump(classifier_64, 'svm_classifier_64.pkl')
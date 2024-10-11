from path import COCO_DIR, GAN_64x64_DIR, STABLE_64x64_DIR, SVM_CHECKPOINT_64, METRICS_FILE_64
from models.diffusion_model import load_diffusion_pipeline
from models.generator import generate_images_with_gan
from models.vgg16_model import load_vgg16_model
from models.svm_classifier import load_svm_classifier
from utils.dataset import ImageDataset
from utils.batch_processing import process_images_in_batches
from torch.utils.data import DataLoader, Subset

# Carica il modello Stable Diffusion
pipeline = load_diffusion_pipeline()

# Carica il modello GAN e genera immagini
generate_images_with_gan()

# Carica il modello VGG16
vgg16 = load_vgg16_model()

# Carica il classificatore SVM
svm_classifier = load_svm_classifier(SVM_CHECKPOINT_64)

# Dataset e DataLoader per COCO e immagini generate
dataset_coco_64 = ImageDataset(COCO_DIR, 64)
dataloader_coco_64 = DataLoader(Subset(dataset_coco_64, range(1000)), batch_size=128, shuffle=False)

dataset_gan_64 = ImageDataset(GAN_64x64_DIR, 64)
dataloader_gan_64 = DataLoader(Subset(dataset_gan_64, range(1000)), batch_size=128, shuffle=False)

dataset_stable_64 = ImageDataset(STABLE_64x64_DIR, 64)
dataloader_stable_64 = DataLoader(Subset(dataset_stable_64, range(1000)), batch_size=128, shuffle=False)

# Processa immagini e calcola metriche
process_images_in_batches(dataloader_coco_64, dataloader_gan_64, dataloader_stable_64, vgg16, svm_classifier, METRICS_FILE_64)

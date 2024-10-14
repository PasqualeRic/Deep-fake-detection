import os
import torch
from config import *
from data_loader import create_dataloader, create_dataloader_with_compression
from generator import load_generator
from metrics import calculate_classification_metrics
from vgg16_classifier import load_vgg16_model, extract_features, load_svm_classifier
from setup_env import setup_environment, load_stable_diffusion_pipeline

def main():
    # Setup environment and load Stable Diffusion pipeline
    device, dtype = setup_environment()
    pipeline = load_stable_diffusion_pipeline(device, dtype)

    # Load VGG16 model and SVM classifiers
    vgg16_model = load_vgg16_model(device)
    svm_64 = load_svm_classifier(SVM_CHECKPOINT_64)
    svm_256 = load_svm_classifier(SVM_CHECKPOINT_256)
    svm_mixed = load_svm_classifier(SVM_CHECKPOINT_MIXED)

    # Dataset and Dataloader for COCO images without resizing
    dataloader_coco_original = create_dataloader(COCO_IMAGES_DIR, batch_size=128, img_size=None)

    # Dataloaders for COCO images with resizing
    dataloader_coco_64 = create_dataloader(COCO_IMAGES_DIR, batch_size=128, img_size=64)
    dataloader_coco_256 = create_dataloader(COCO_IMAGES_DIR, batch_size=128, img_size=256)

    # Dataloaders for GAN and Stable Diffusion generated images
    dataloader_gan_64 = create_dataloader(GAN_IMAGES_DIR_64, batch_size=128, img_size=64)
    dataloader_gan_256 = create_dataloader(GAN_IMAGES_DIR_256, batch_size=128, img_size=256)
    dataloader_stable_64 = create_dataloader(STABLE_IMAGES_DIR_64, batch_size=128, img_size=64)
    dataloader_stable_256 = create_dataloader(STABLE_IMAGES_DIR_256, batch_size=128, img_size=256)

    # Test 1: COCO vs GAN 64x64
    print("Processing COCO vs GAN 64x64...")
    process_images_in_batches(dataloader_coco_64, dataloader_gan_64, vgg16_model, svm_64, '/kaggle/working/metrics_coco_gan_6464.txt')

    # Test 2: COCO vs GAN 256x256
    print("Processing COCO vs GAN 256x256...")
    process_images_in_batches(dataloader_coco_256, dataloader_gan_256, vgg16_model, svm_256, '/kaggle/working/metrics_coco_gan_256256.txt')

    # Test 3: COCO vs Stable Diffusion 64x64
    print("Processing COCO vs Stable Diffusion 64x64...")
    process_images_in_batches(dataloader_coco_64, dataloader_stable_64, vgg16_model, svm_64, '/kaggle/working/metrics_coco_stable_6464.txt')

    # Test 4: COCO vs Stable Diffusion 256x256
    print("Processing COCO vs Stable Diffusion 256x256...")
    process_images_in_batches(dataloader_coco_256, dataloader_stable_256, vgg16_model, svm_256, '/kaggle/working/metrics_coco_stable_256256.txt')

    # Test 5: COCO (no resize) vs GAN 64x64
    print("Processing COCO (no resize) vs GAN 64x64...")
    process_images_in_batches(dataloader_coco_original, dataloader_gan_64, vgg16_model, svm_mixed, '/kaggle/working/metrics_coco_noresize_gan64.txt')

    # Test 6: COCO (no resize) vs Stable Diffusion 256x256
    print("Processing COCO (no resize) vs Stable Diffusion 256x256...")
    process_images_in_batches(dataloader_coco_original, dataloader_stable_256, vgg16_model, svm_mixed, '/kaggle/working/metrics_coco_noresize_stable256.txt')

    print("Processing complete.")

if __name__ == "__main__":
    main()

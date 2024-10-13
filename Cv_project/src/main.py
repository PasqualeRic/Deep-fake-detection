import os
import torch
from config import *
from data_loader import create_dataloader_with_compression, create_dataloader
from generator import load_generator
from metrics import calculate_inception_score, save_fis_results, calculate_fid, calculate_classification_metrics
from vgg16_classifier import load_vgg16_model, extract_features, load_svm_classifier
from setup_env import setup_environment, load_stable_diffusion_pipeline
from PIL import Image
import json

def main():
    # Setup environment and load Stable Diffusion pipeline
    device, dtype = setup_environment()
    pipeline = load_stable_diffusion_pipeline(device, dtype)

    # Load COCO annotations
    coco_data = load_coco_annotations(COCO_ANNOTATION_FILE)

    # Map category IDs to category names
    categories = {cat['id']: cat['name'] for cat in coco_data['categories']}

    # Extract image annotations
    image_annotations = {}
    for annotation in coco_data['annotations']:
        img_id = annotation['image_id']
        category_name = categories[annotation['category_id']]
        
        if img_id in image_annotations:
            image_annotations[img_id].append(category_name)
        else:
            image_annotations[img_id] = [category_name]

    # Select images (from 2000 to 4000 as in the notebook)
    sample_ids = list(image_annotations.keys())[2000:4000]  # Seleziona 2000 immagini

    # Generate and save 2000 images using Stable Diffusion with COCO prompts
    os.makedirs(STABLE_IMAGES_DIR_64, exist_ok=True)
    os.makedirs(STABLE_IMAGES_DIR_256, exist_ok=True)

    for i, img_id in enumerate(sample_ids, start=1):
        try:
            prompt = ", ".join(image_annotations[img_id])
            generated_image = generate_image_from_prompt(pipeline, prompt)
            save_generated_images(generated_image, STABLE_IMAGES_DIR_64, STABLE_IMAGES_DIR_256, i)
            print(f"Images {i}/2000 generated and saved using Stable Diffusion.")
        except Exception as e:
            print(f"Error generating image ID {img_id}: {e}")

    # Load GAN generator and generate exactly 2000 fake images as in the notebook
    generator = load_generator('/path/to/checkpoint_generator.pth', device)
    generate_fake_images_from_generator(generator, device, num_images=2000)  # Generate 2000 images

    # Calculate IS and FIS metrics
    real_images_loader = create_dataloader(COCO_IMAGES_DIR, img_size=64)
    generated_images_loader = create_dataloader(STABLE_IMAGES_DIR_64, img_size=64)

    # Load VGG16 model and SVM classifier
    vgg16_model = load_vgg16_model(device)
    svm_classifier = load_svm_classifier(SVM_CHECKPOINT_64)

    # Extract features
    real_features = extract_features(next(iter(real_images_loader))[0], vgg16_model, device)
    generated_features = extract_features(next(iter(generated_images_loader))[0], vgg16_model, device)

    # Classification with SVM
    print("Calculating classification metrics using the SVM classifier...")
    y_true = [1] * len(real_features) + [0] * len(generated_features)  # Real = 1, Generated = 0
    y_pred = svm_classifier.predict(np.vstack([real_features, generated_features]))

    # Save classification metrics (Accuracy, Precision, Recall, F1)
    metrics_save_path = os.path.join(RESULTS_DIR, 'classification_metrics.txt')
    calculate_classification_metrics(y_true, y_pred, metrics_save_path)

    # Optionally calculate FID
    fid_score = calculate_fid(real_images_loader, generated_images_loader, device)
    print(f"FID Score: {fid_score}")

if __name__ == "__main__":
    main()

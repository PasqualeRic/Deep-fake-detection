Image Generation and Classification Project with GAN and Stable Diffusion
This project implements a system for generating images using GAN and Stable Diffusion, comparing them with real images through a VGG16 model to extract features and an SVM classifier to predict whether images are real or generated. The project also includes the calculation of accuracy, precision, recall, and F1 score metrics.

Project Structure
models/

diffusion_model.py: Contains the code to load and use the Stable Diffusion pipeline.
generator.py: Contains the code to define the GAN model and generate images.
svm_classifier.py: Contains the code to load the pre-trained SVM classifier.
vgg16_model.py: Contains the code to load the VGG16 model used for feature extraction.
utils/

dataset.py: Defines a class for creating a custom dataset to load and preprocess images.
feature_extraction.py: Contains a function to extract features from images using a pre-trained model (VGG16).
metrics.py: Contains functions to calculate and save accuracy, precision, recall, and F1 score metrics.
batch_processing.py: Contains a function to process real and generated images in batches, extracting features and comparing them for classification.
path.py: A file that contains the main paths used in the project for datasets, checkpoints, and results.

main.py: The main file to run the entire process of image generation, feature extraction, and metric calculation.

Features
Image Generation:

Stable Diffusion: Generates 2000 images based on text prompts extracted from the COCO dataset.
GAN: Generates 2000 images using a pre-trained GAN generator model.
Metric Calculation:

Inception Score (IS): Evaluates the quality and diversity of generated images.
Fréchet Inception Distance (FID): Measures how similar the generated images are to real images.
Feature Importance Score (FIS): Compares feature distributions between real and generated images.
Classification Metrics: Accuracy, Precision, Recall, and F1 Score for binary classification (real vs generated) using a pre-trained SVM classifier and VGG16 for feature extraction.
Image Compression:

The project allows generating compressed datasets with JPEG compression at various quality levels (40%, 60%, 80%).

Dependencies
The project relies on several Python packages. You can install the required dependencies with the following command:

bash:
pip install -r requirements.txt



Dependencies include:
torch
diffusers
Pillow
scikit-learn
joblib
numpy
torchvision
The specific versions of the libraries are listed in the requirements.txt file.
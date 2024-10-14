import os

BASE_PATH = os.getcwd()

COCO_ANNOTATION_FILE = os.path.join(BASE_PATH, 'data', 'annotations', 'instances_train2017.json')
COCO_IMAGES_DIR = os.path.join(BASE_PATH, 'data', 'train2017')

GAN_IMAGES_DIR_64 = os.path.join(BASE_PATH, 'Immagini_gan', '64x64')
GAN_IMAGES_DIR_256 = os.path.join(BASE_PATH, 'Immagini_gan', '256x256')
STABLE_IMAGES_DIR_64 = os.path.join(BASE_PATH, 'Immagini_stable', '64x64')
STABLE_IMAGES_DIR_256 = os.path.join(BASE_PATH, 'Immagini_stable', '256x256')

SVM_CHECKPOINT_64 = os.path.join(BASE_PATH, 'models', 'svm_classifier_64.pkl')
SVM_CHECKPOINT_256 = os.path.join(BASE_PATH, 'models', 'svm_classifier_256.pkl')
SVM_CHECKPOINT_MIXED = os.path.join(BASE_PATH, 'models', 'svm_classifier_mixed.pkl')

RESULTS_DIR = os.path.join(BASE_PATH, 'results')

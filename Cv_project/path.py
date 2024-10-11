# Percorsi principali per i dati e i modelli
COCO_DIR = '/path/to/coco'
GAN_64x64_DIR = '/path/to/gan_images/64x64'
STABLE_64x64_DIR = '/path/to/stable_images/64x64'

# Percorsi per il caricamento dei checkpoint
SVM_CHECKPOINT_64 = '/path/to/checkpoints/svm_classifier_64.pkl'
SVM_CHECKPOINT_256 = '/path/to/checkpoints/svm_classifier_256.pkl'
GAN_CHECKPOINT = '/path/to/checkpoints/checkpoint_generator.pth'

# Percorsi per i risultati delle metriche
METRICS_FILE_64 = '/path/to/results/metrics_combined_64x64.txt'
METRICS_FILE_256 = '/path/to/results/metrics_combined_256x256.txt'

# Percorsi per le directory delle immagini generate
SAVE_DIRECTORY_GAN = '/path/to/saved_images_gan'
SAVE_DIRECTORY_STABLE = '/path/to/saved_images_stable'

# Percorsi per il dataset COCO e annotazioni
COCO_ANNOTATION_FILE = '/path/to/coco/annotations/instances_train2017.json'
COCO_IMAGES_DIR = '/path/to/coco/images/train2017'

# Altri percorsi di salvataggio
RESULTS_DIR = '/path/to/results'
IS_FIS_RESULTS_DIR = '/path/to/results/IS_FIS_results'

import torch
from diffusers import DiffusionPipeline
from torchvision import models, transforms
from PIL import Image
import os
from pycocotools.coco import COCO
import numpy as np
from sklearn.svm import SVC
import joblib

# this function load the last available checkpoint
def load_checkpoint(checkpoint_dir):
    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith('svm_classifier_checkpoint')]
    if checkpoints:
        latest_step = max([int(f.split('_')[-1].split('.')[0]) for f in checkpoints])
        #these are the paths to the folder in my drive
        model_path = os.path.join(checkpoint_dir, f'svm_classifier_checkpoint_{latest_step}.pkl')
        features_path = os.path.join(checkpoint_dir, f'features_checkpoint_{latest_step}.npy')
        labels_path = os.path.join(checkpoint_dir, f'labels_checkpoint_{latest_step}.npy')

        # load the SVM, features and labels
        classifier = joblib.load(model_path)
        features_list = np.load(features_path, allow_pickle=True).tolist()
        labels = np.load(labels_path, allow_pickle=True).tolist()

        print(f"Checkpoint successfully loaded {latest_step}")
        return classifier, features_list, labels, latest_step
    else:
        return None, [], [], 0

#The train2017 and instances_train2017_subset are a subsets of COCO's datset, they contain only 2000 images
coco_root = '/content/drive/MyDrive/dataset_ridotto_test/train2017'
coco_annotation_file = '/content/drive/MyDrive/dataset_ridotto_test/instances_train2017_subset.json'
coco = COCO(coco_annotation_file)

# check if I have a GPU available 
device = "cuda" if torch.cuda.is_available() else "cpu"

dtype = torch.float16 if device == "cuda" else torch.float32

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1", torch_dtype=dtype)

# accelerate improve the performance
try:
    pipeline.enable_attention_slicing()
    print("Accelerate abilitato per migliorare le prestazioni.")
except ImportError:
    print("Attenzione: 'accelerate' non trovato.")

pipeline.to(device)

#take the data given from the dataset and generate the image with stable diffusion
def generate_image(prompt):
    image = pipeline(prompt, num_inference_steps=8).images[0]
    return image.convert("RGB")

#
def extract_vgg_features(image, model):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image).unsqueeze(0).to(device)  # add the dimension batch and transfer to the GPU
    with torch.no_grad():
        features = model(input_tensor)
    return features.flatten().cpu().numpy()

# Funzione per salvare le immagini fake
def save_image(image, save_dir, img_id):
    os.makedirs(save_dir, exist_ok=True)  # Crea la cartella se non esiste
    save_path = os.path.join(save_dir, f"fake_image_{img_id}.jpg")
    image.save(save_path, format="JPEG")
    print(f"Immagine salvata in: {save_path}")

# this function save a checkpoint of the SVM model, beacuse in google colab I can't process more than 500 images with a GPU, so I save and restert the training from 500
def save_checkpoint(classifier, features_list, labels, step):
    checkpoint_dir = "/content/drive/MyDrive/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # save the SVM model
    model_path = os.path.join(checkpoint_dir, f'svm_classifier_checkpoint_{step}.pkl')
    joblib.dump(classifier, model_path)

    features_path = os.path.join(checkpoint_dir, f'features_checkpoint_{step}.npy')
    labels_path = os.path.join(checkpoint_dir, f'labels_checkpoint_{step}.npy')

    np.save(features_path, features_list)
    np.save(labels_path, labels)

    print(f"Checkpoint saved: {model_path}")

# Load a VGG16 pre-trained model and remove the last layer
vgg16 = models.vgg16(pretrained=True)
vgg16.classifier = vgg16.classifier[:-1]  # remove the last layer
vgg16.eval().to(device)  #put in eval modality and tansfer to the GPU

#I save the fake image generated by stable diffusion
save_dir = "/content/drive/MyDrive/fake_image"
checkpoint_dir = "/content/drive/MyDrive/checkpoints" 

# restart training from the checkpoist if it is possible 
classifier, features_list, labels, last_step = load_checkpoint(checkpoint_dir)

# if none checkpoint is available start from zero
if classifier is None:
    classifier = SVC(kernel='linear')
    features_list = []
    labels = []
    last_step = 0

#thake the coco's images
image_ids = coco.getImgIds()
sample_ids = image_ids[last_step:]  # last step is the last step of the checkpoint

checkpoint_interval = 500  # every 500 images will save a checkpoint
step = last_step

for img_id in sample_ids:
    img_data = coco.imgs[img_id]
    img_path = os.path.join(coco_root, img_data['file_name'])
    if(step >= 1000): #I choose to stop the iteration at 1000 images, beacuse it is to much computationally expensive train and generate for each image
        break
  
    try:
        #take the real image
        image_real = Image.open(img_path).convert("RGB")
        real_features = extract_vgg_features(image_real, vgg16)
        features_list.append(real_features)
        labels.append(1)  # in labels append 1 for the real images

        # get the category for each image to generate fake images, in coco we have the annotations for each image not the image
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        categories = [coco.cats[ann['category_id']]['name'] for ann in anns]
        prompt = ", ".join(categories)

        
        image_fake = generate_image(prompt)

        save_image(image_fake, save_dir, img_id)

        fake_features = extract_vgg_features(image_fake, vgg16)
        features_list.append(fake_features)
        labels.append(0)  # in lapels append 0 for the fake image

        step += 1

        #save the checkpoint each 500 images
        if step % checkpoint_interval == 0:
            classifier.fit(features_list, labels) 
            save_checkpoint(classifier, features_list, labels, step)

    except Exception as e:
        print(f"Errore nell'elaborazione dell'immagine ID {img_id}: {e}")

# training the model on all the extracted features and labels
classifier.fit(features_list, labels)

# save the final model
joblib.dump(classifier, 'svm_classifier.pkl')
print("Modello SVM salvato come 'svm_classifier.pkl'")


#from google.colab import drive
#drive.mount('/content/drive')
# pip install diffusers
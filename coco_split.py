import torch
from pycocotools.coco import COCO
import os
import shutil
import json
import random

# Percorso al tuo dataset COCO
coco_root = '/Users/pasqualericciulli/Downloads/train2017'  # Cambia questo con il tuo percorso
coco_annotation_file = os.path.join('/Users/pasqualericciulli/Downloads/annotations/instances_train2017.json')
coco = COCO(coco_annotation_file)

# Funzione per copiare le immagini nelle nuove cartelle
def copy_images(image_ids, destination_folder):
    for img_id in image_ids:
        img_data = coco.imgs[img_id]
        img_file_name = img_data['file_name']
        src_path = os.path.join(coco_root, img_file_name)  # Rimuovi 'train2017'
        dest_path = os.path.join(destination_folder, img_file_name)  # Nuovo percorso dell'immagine
        shutil.copy(src_path, dest_path)  # Copia l'immagine nella nuova cartella

# Funzione per creare un file di annotazione
def create_annotation_file(image_ids, destination_file):
    annotations = {
        'images': [],
        'annotations': [],
        'categories': coco.dataset['categories']
    }
    
    for img_id in image_ids:
        img_data = coco.imgs[img_id]
        annotations['images'].append(img_data)
        
        # Aggiungi le annotazioni corrispondenti
        ann_ids = coco.getAnnIds(imgIds=img_id)
        for ann_id in ann_ids:
            annotations['annotations'].append(coco.anns[ann_id])
    
    # Salva il file JSON
    with open(destination_file, 'w') as f:
        json.dump(annotations, f)

# Specifica un nuovo percorso per la cartella di suddivisione
new_coco_root = '/Users/pasqualericciulli/Downloads/coco_split'  # Cambia questo con il tuo percorso desiderato

# Crea cartelle per salvare i dataset suddivisi
os.makedirs(os.path.join(new_coco_root, 'train'), exist_ok=True)
os.makedirs(os.path.join(new_coco_root, 'eval'), exist_ok=True)
os.makedirs(os.path.join(new_coco_root, 'test'), exist_ok=True)

# Esempio di utilizzo: prendi alcune immagini dal dataset
image_ids = coco.getImgIds()
random.shuffle(image_ids)  # Mescola gli ID delle immagini

# Limita a sole 10 immagini per ogni sottoinsieme
num_images = min(10, len(image_ids))  # Usa il numero minore tra 10 e il numero totale di immagini
train_ids = image_ids[:int(num_images * 0.7)]   # 70% per il training
eval_ids = image_ids[int(num_images * 0.7):int(num_images * 0.85)]  # 15% per la valutazione
test_ids = image_ids[int(num_images * 0.85):]  # 15% per il test

# Copia le immagini nei rispettivi set
copy_images(train_ids, os.path.join(new_coco_root, 'train'))
copy_images(eval_ids, os.path.join(new_coco_root, 'eval'))
copy_images(test_ids, os.path.join(new_coco_root, 'test'))

# Crea i file di annotazione per il nuovo percorso
os.makedirs(os.path.join(new_coco_root, 'annotations'), exist_ok=True)  # Crea la cartella per le annotazioni
create_annotation_file(train_ids, os.path.join(new_coco_root, 'annotations', 'instances_train.json'))
create_annotation_file(eval_ids, os.path.join(new_coco_root, 'annotations', 'instances_eval.json'))
create_annotation_file(test_ids, os.path.join(new_coco_root, 'annotations', 'instances_test.json'))

print("Dataset COCO suddiviso in train, eval e test e copiato in:", new_coco_root)

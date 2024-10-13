# Image Generation and Classification Project with GAN, Stable Diffusion and VGG16
This project implements a system to generate general images using a **GAN** and **Stable Diffusion**, comparing them with real images taken from COCO dataset.
To predict wheter an image is real or generated, we use a **VGG16**, of which we have removed the last layer and replaced it with an **SVM**, wich we have re-trained to allow us to predict the class (real or generated), of the images.

**Authors**: Pasquale Ricciulli, Francesco Conforti

## Datase
Il COCO (Common Objects in Context) è uno dei dataset più utilizzati e completi per lo sviluppo di modelli di visione artificiale. È stato progettato per supportare diverse attività di computer vision, come il riconoscimento di oggetti, la segmentazione, la didascalizzazione di immagini, e la comprensione contestuale.

Caratteristiche principali del dataset COCO:
Immagini annotate: Il dataset contiene oltre 330.000 immagini ad alta risoluzione, di cui più di 200.000 sono annotate con didascalie dettagliate.
Oggetti e categorie: COCO include oltre 80 categorie di oggetti comuni come persone, animali, veicoli, utensili, e strumenti presenti nelle immagini. Ciascuna immagine può contenere uno o più oggetti annotati.
Segmentazione e bounding box: Oltre al riconoscimento degli oggetti, COCO fornisce annotazioni di segmentazione per ogni oggetto, il che significa che gli oggetti vengono delineati in modo preciso all'interno dell'immagine (oltre ai tradizionali bounding box).
Didascalie (captions): Una parte importante del dataset contiene didascalie in linguaggio naturale che descrivono il contenuto delle immagini, utile per task come la generazione automatica di descrizioni di immagini o la visual question answering (VQA).
Scene complesse: Le immagini di COCO non contengono solo oggetti isolati, ma spesso presentano oggetti in contesti reali, con sovrapposizioni e interazioni tra gli oggetti, il che rende il dataset molto utile per capire la relazione tra oggetti in ambienti complessi.

## Stable Diffusion
Stable diffusion is a pre-trained model "stabilityai/stable-diffusion-2-1" that takes annotations from the coco dataset and generates an image with them.
This is an example of image generated by Stable diffusion:

<img src='imgs/fake_image_75748.jpg' width="200px"/>


## GAN
The GAN model, on the other hand, was created from scratch by us, has 5 convolutional layers, has a batch size of 128, generates the output images with a resolution of 64x64 and it was trained on coco images over 100 epochs.
these are the result of 1st epoch and 100th epoch:

<p float="left">
  <img src='imgs/4.png' width="300px"/>
  <img src='imgs/final_generated_images.png' width="300px"/>
</p>

Di seguito riportato il grafico contenente le loss che abbiamo salvato durante la fase di addestramento:
<img src='imgs/loss_plot.png' width="300px"/>
## VGG16 classifier
This model is used for feature extraction and as a classifier, we have used a pre-trained model for feature extraction, but we have trained an SVM as the last layer of VGG16 to make predictions, in this work, we traind three SVM models, one for **normal images**, one for **64x64 images** and one for **256x256 images**, each SVM model takes as input 3 images, one real with label 1 and two generated images (one for GAN and one for stable) with label 0.
This is an example with original, downsampling to 64x64 and upsampling to 256x256:
<p float="left">
  <img src='imgs/original/real_image_real_image_248242.jpg.jpg' width="200px"/>
  <img src='imgs/original/image_64x64real_image_248242.jpg.jpg' width="200px"/>
  <img src='imgs/original/image_256x256real_image_248242.jpg.jpg' width="200px"/>
</p>

## Test
In this section we want to find out which is the best model to generate the images.

|**Test type**                        |
|-------------------------------------|
|**Gan 256x256**                      |
|**Stable 256x256**                   |
|**Gan 64x64**                        |
|**Stable 64x64**                     | 
|**Combined 64x64** **Compession 80%**| 
|**Combined 64x64 Compression 60%**   |
|**Combined 64x64 Compression 40%**   | 
|**Combined 64x64**                   |


**GAN 256x256**: Questo modello utilizza immagini generate da una GAN a risoluzione 256x256. La maggiore risoluzione potrebbe teoricamente consentire al classificatore di identificare dettagli più sottili e caratteristiche distintive delle immagini.

**Stable 256x256**:Questo modello utilizza immagini generate da Stable Diffusion a risoluzione 256x256. Stable Diffusion è noto per la sua capacità di generare immagini di alta qualità e realismo.

**Gan 64x64**: Descrizione: Utilizza immagini generate da una GAN a risoluzione 64x64. La bassa risoluzione può limitare la capacità del modello di riconoscere dettagli importanti.
**Stable 64x64**: Le immagini sono generate da Stable Diffusion a risoluzione 64x64. Anche se Stable Diffusion è in grado di generare immagini di alta qualità, la risoluzione ridotta può influire sulla capacità del classificatore di identificare differenze significative.

**Combined 64x64 compression 80%**:  Questo modello contine imamgini della stable, immagini della gan e immagini originali e  combina la bassa risoluzione con una compressione del 80%. La compressione potrebbe ridurre ulteriormente la qualità delle immagini.

**Combined 64x64 compression 60%**:  Modello simile al precedente, ma con compressione ridotta al 60%. Questo potrebbe preservare un po' più di qualità dell'immagine.

**Combined 64x64 compression 40%**: Ancora un altro modello che utilizza immagini a bassa risoluzione, ma con compressione al 40%.

**Combined 64x64**: Questo modello utilizza immagini a 64x64 senza compressione.

The metrics used are **accuracy**, **recall**, **f1-score** and **precision**.

## Results
The results are shown in the table below:
|**Test type**                        | **Accuracy** | **Precision** | **Recall**  |**F1-Score**|
|-------------------------------------|--------------|---------------|-------------|------------|
|**Gan 256x256**                      |  0.8040      |  0.7565       |  0.8206     |  0.8206    |
|**Stable 256x256**                   |  0.8990      |  0.9010       |  0.8987     |  0.8987    |
|**Gan 64x64**                        |  0.5543      |  0.5328       |  0.8805     |  0.6639    |
|**Stable 64x64**                     |  0.5543      |  0.5328       |  0.8805     |  0.6639    |
|**Combined 64x64** **Compession 80%**|  0.7348      |  0.6825       |  0.8780     |  0.7680    |
|**Combined 64x64 Compression 60%**   |  0.7145      |  0.6616       |  0.8780     |  0.7546    |
|**Combined 64x64 Compression 40%**   |  0.6993      |  0.6468       |  0.8780     |  0.7449    |
|**Combined 64x64**                   |  0.6583      |  0.6096       |  0.8805     |  0.7204    |

**GAN 256x256**: Presenta buone metriche di accuratezza, precisione, richiamo e F1-score, suggerendo che il modello riesce a discriminare tra immagini reali e fake con una certa efficacia.

**Stable 256x256**: Mostra risultati migliori rispetto al modello GAN a 256x256, con elevate metriche di accuratezza e precisione. Ciò indica che il classificatore riesce a distinguere le immagini fake in modo più affidabile.

**Gan 64x64**: Mostra risultati significativamente inferiori rispetto agli altri modelli, con bassa precisione e recall. Questo suggerisce che le immagini a bassa risoluzione non forniscono informazioni sufficienti per una classificazione efficace.

**Stable 64x64**: Le metriche sono simili a quelle della GAN 64x64, mostrando che la bassa risoluzione limita l'efficacia di questo modello.

**Combined 64x64 compression 80%**: Mostra miglioramenti rispetto ai modelli a 64x64 senza compressione, suggerendo che l'inclusione di immagini di tutti i modelli aiuta a migliorare la capacità di classificazione, anche se rimane insufficiente rispetto ai modelli a 256x256.

**Combined 64x64 compression 60%**: I risultati mostrano un ulteriore miglioramento rispetto ai modelli gan e stable 64x64.

**Combined 64x64 compression 40%**: Questo modello presenta le peggiori metriche tra quelli a 64x64 con compressione, indicando che mantenere una minore qualità dell'immagine, anche in dimensioni ridotte, può peggiorare le prestazioni del classificatore.

**Combined 64x64**: Presenta le metriche di prestazione più basse tra i modelli combined , confermando che senza compressione, le limitazioni della risoluzione bassa influiscono negativamente sulla capacità del modello di distinguere le immagini.



## Structure of the project
- `Test/`
  - `Notebook/`
    -`Test.ipynb`:This file could be uploaded to kaggle to work.
  - `src/`
    - `config.py`:Configuration file that defines file paths and directories, including COCO annotations, directories for generated images, and SVM model checkpoints​.
    - `data_loader.py`:Defines custom datasets for loading and preprocessing images, with or without compression, and creates DataLoaders for batch processing of images.
    - `generator.py`:Contains the definition and loading of the GAN generator model used to generate images from a latent vector. It also includes a function to load a pre-trained generator model from a checkpoint.
    - `main.py`:The main script that sets up the environment, loads the GAN and Stable Diffusion models, generates images, and calculates evaluation metrics like FID and IS. It also handles COCO annotations and performs classification using an SVM classifier​.
    - `metrics.py`:Implements functions to calculate the Inception Score (IS), Fréchet Inception Distance (FID), and classification metrics (Accuracy, Precision, Recall, F1 Score). It also includes a function to save the metric results.
    - `setup_env.py`:Manages the environment setup by checking for GPU availability and loading the Stable Diffusion pipeline for image generation​.
    - `stable_diffusion_pipeline.py`: Provides functions to load the Stable Diffusion pipeline and generate images based on text prompts​.
    - `vgg16_classifier`:Contains functions to load a pre-trained VGG16 model for image feature extraction and to load a pre-trained SVM classifier.
      
- `VGG16/`
  - `training-vgg16.ipynb`: This file could be uploaded to kaggle to work.
  - `training_VGG16_64_256`: This file contains the code for training the SVM model with image sizes 64x64 and 256x256 from coco.
  - `training_VGG16_normal.py`: This file contains the code for generating the fake images with stable diffusion, for creating the dataset for training and testing and also for training the model on images without scaling.

- `GAN/`
  - `Generator.py`: This file contains the code for generating the fake images with GAN.
  - `training_GAN.py`: This file contains the code for training the GAN model with image from coco.
  - `gan.ipynb`: This file could be uploaded to kaggle to work
## Installation

1. Clone the repository.
   ```bash
   git clone https://github.com/PasqualeRic/Deep-fake-detection.git
   cd Deep-fake-detection


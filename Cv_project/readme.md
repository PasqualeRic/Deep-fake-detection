# Progetto di Generazione e Classificazione di Immagini con GAN e Stable Diffusion

Questo progetto implementa un sistema per generare immagini utilizzando GAN e Stable Diffusion, confrontandole con immagini reali attraverso un modello VGG16 per estrarre le caratteristiche e un classificatore SVM per predire la classe delle immagini (reale vs. generata). Include anche il calcolo di metriche di accuratezza, precisione, richiamo e F1 score.

## Struttura del Progetto

- `models/`
  - `diffusion_model.py`: Contiene il codice per caricare e utilizzare la pipeline di Stable Diffusion.
  - `generator.py`: Contiene il codice per definire il modello generativo GAN e generare immagini.
  - `svm_classifier.py`: Contiene il codice per caricare il classificatore SVM.
  - `vgg16_model.py`: Contiene il codice per caricare il modello VGG16 utilizzato per estrarre le caratteristiche.
  
- `utils/`
  - `dataset.py`: Definisce una classe per creare un dataset personalizzato per caricare e preprocessare le immagini.
  - `feature_extraction.py`: Contiene una funzione per estrarre le caratteristiche dalle immagini utilizzando un modello pre-addestrato.
  - `metrics.py`: Contiene una funzione per calcolare e salvare le metriche di accuratezza, precisione, richiamo e F1 score.
  - `batch_processing.py`: Contiene una funzione per processare le immagini reali e generate in batch, estraendo le caratteristiche e confrontandole.

- `path.py`: File che contiene i percorsi principali utilizzati nel progetto per i dataset, checkpoint e risultati.

- `main.py`: Il file principale per lanciare il processo completo di generazione, estrazione delle caratteristiche e calcolo delle metriche.

## Dipendenze

Il progetto richiede le seguenti librerie:

- `torch`
- `diffusers`
- `Pillow`
- `scikit-learn`
- `joblib`
- `numpy`
- `torchvision`

Le versioni specifiche delle librerie sono incluse nel file `requirements.txt`.

## Installazione

1. Clona il repository.
   ```bash
   git clone https://github.com/tuo-username/tuo-repository.git
   cd tuo-repository

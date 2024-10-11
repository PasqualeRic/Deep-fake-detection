import joblib

def load_svm_classifier(path):
    """
    Carica il classificatore SVM da un file di checkpoint.

    Args:
    - path (str): Percorso al file SVM.

    Returns:
    - classifier: Classificatore SVM caricato.
    """
    return joblib.load(path)

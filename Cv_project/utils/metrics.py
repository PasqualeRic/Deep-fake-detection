from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def calculate_metrics(y_true, y_pred, save_path):
    """
    Calcola le metriche di accuratezza, precisione, richiamo e F1 e salva i risultati su file.

    Args:
    - y_true (list): Valori reali delle etichette.
    - y_pred (list): Predizioni del modello.
    - save_path (str): Percorso dove salvare i risultati delle metriche.
    """
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    with open(save_path, 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

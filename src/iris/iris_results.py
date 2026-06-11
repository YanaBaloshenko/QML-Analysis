import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, mean_squared_error, mean_absolute_error

from qiskit_machine_learning.algorithms import VQC, VQR, QSVC, QSVR

def report(folder):
    with open(f"{folder}/metadata.json", "r") as f:
        data = json.load(f)

    filename = data["filename"]
    ml_type = data.get("ml_type", filename.split('_')[0])

    if ml_type == "vqc":
        ml = VQC.from_dill(f"{folder}/{filename}.model")
    elif ml_type == "vqr":
        ml = VQR.from_dill(f"{folder}/{filename}.model")
    elif ml_type == "qsvc":
        ml = QSVC.from_dill(f"{folder}/{filename}.model")
    elif ml_type == "qsvr":
        ml = QSVR.from_dill(f"{folder}/{filename}.model")

    d_file = data["d_file"]
    sets = np.load(f"dataset/{d_file}/{d_file}.npz")

    train_features = sets['train_features']
    train_labels = sets['train_labels']
    test_features = sets['test_features']
    test_labels = sets['test_labels']

    test_score = ml.score(test_features, test_labels)
    train_score = ml.score(train_features, train_labels)
 
    # saving report
    with open(f"{folder}/{filename}_report.txt", "w") as f:
        f.write(f"Model {type(ml).__name__} | Job Id {data.get('job_id', 'N/A')}\n")
        f.write("\n--- Dataset info ---\n")
        f.write(f"Data file used: {d_file}\n")
        f.write(f"Number of features: {data.get('n_features', 'N/A')}\n")
        f.write(f"Number of records: train - {train_features.shape[0]}, test - {test_features.shape[0]}\n")

        f.write("\n--- Training info ---\n")
        f.write(f"Number of classes: {data.get('num_classes', 'N/A')}\n")
        f.write(f"Training time: {data.get('train_time', 'N/A')} s\n")
        f.write(f"Accuracy on training set: {train_score:.2f}\n")
        f.write(f"Accuracy: {test_score:.2f}\n")
        
        f.write(f"\n--- Optimizer ({data.get('optimizer_name', 'N/A')}) set settings ---\n")
        for key, value in data.get("optimizer_settings", {}).items():
            f.write(f"{key}: {value}\n")

def main():
    folder = "results/ideal/vqc/vqc_data-iris_2-fpca_onehot-enc_150_backend-ideal_time-11062026_0144"
    report(folder)

if __name__ == "__main__":
    main()


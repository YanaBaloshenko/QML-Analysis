import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from qiskit_machine_learning.algorithms import VQC, VQR, QSVC, QSVR

def predict(folder):
    with open(f"{folder}/metadata.json", "r") as f:
        data = json.load(f)

    filename = data["filename"]

    if "ml_type" in data:
        ml_type = str(data["ml_type"])
    else:
        ml_type = str(filename.split('_')[0])

    if ml_type=="vqc":
        ml = VQC.from_dill(f"{folder}/{filename}.model")
    elif ml_type=="vqr":
        ml = VQR.from_dill(f"{folder}/{filename}.model")
    elif ml_type=="qsvc":
        ml = QSVC.from_dill(f"{folder}/{filename}.model")
    elif ml_type=="qsvr":
        ml = QSVR.from_dill(f"{folder}/{filename}.model")
    else:
        raise ValueError("Unknown model type.")

    d_file = data["d_file"]
    sets = np.load(f"dataset/{d_file}/{d_file}.npz")
    num_rec = data.get('num_rec', None)
    test_features = sets['test_features']
    if num_rec is not None:
        test_features = sets['test_features'][:num_rec]
    
    print("Calculating predictions...")
    predictions = ml.predict(test_features)
        
    data["predictions"] = predictions.tolist()
        
    with open(f"{folder}/metadata.json", "w") as f:
        json.dump(data, f, indent=4)
    print("Predictions successfully saved to metadata.json")

    if ml_type == "vqr":
        plt.figure(figsize=(10, 6)) 
        plt.scatter(range(len(predictions)), predictions, color='blue', alpha=0.7, label="VQR Expectation Value")
        plt.axhline(y=0.0, color='red', linestyle='--', label='Decision Boundary (0.0)')
        plt.title("VQR Predictions (Expectation Values) on Hardware")
        plt.xlabel("Test Record Index")
        plt.ylabel("Measured Expectation Value")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.savefig(f"{folder}/{filename}_vqr_scatter.png", dpi=300)
        plt.close()

def main():
    folder = 'results/ideal/vqc/vqc_data-kdd_3.14-scale_5-fpca_onehot-enc_240_backend-ideal_time-09062026_1650'
    predict(folder)

if __name__ == "__main__":
    main()

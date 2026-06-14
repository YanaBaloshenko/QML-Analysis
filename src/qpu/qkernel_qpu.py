import numpy as np
import datetime
import time
import os
import json

import qpu_algorithm

def results_save(ml_type, ml, sampler, bcknd, options, folder, train_time, n_features, d_file, filename, test_features):
    num_shots = None
    if bcknd != "ideal":
        num_shots = sampler.options.default_shots

    print(f"\nSaving data to {folder}...")
    metadata = {
        "ml_type": ml_type,
        "train_time": train_time,
        "n_features": n_features,
        "d_file": d_file,
        "num_shots": num_shots,
        "filename": filename,
        "bcknd": bcknd,
        "num_qubits": ml.quantum_kernel.feature_map.num_qubits,
        "support_vectors": len(ml.support_vectors_) if hasattr(ml, 'support_vectors_') else 0,
        "options": options
    }

    print("Calculating predictions...")
    predictions = ml.predict(test_features)
    metadata.update({"predictions": predictions.tolist()})
    print("Predictions successfully saved.")

    with open(f"{folder}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

def training(ml_type, bcknd, d_file):
    d_path = f"dataset/{d_file}"
    date = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    r_folder = f"results/qpu/{ml_type}"
    filename = f"{ml_type}_data-{d_file}_backend-{bcknd}_time-{date}"
    folder = f"{r_folder}/{filename}"
    os.makedirs(folder, exist_ok=True)

    ################## data read ##################
    data = np.load(f"{d_path}/{d_file}.npz")
    train_features = data['train_features']
    train_labels = data['train_labels']
    test_features = data['test_features']
    n_features = train_features.shape[1]

    ################## training and saving results ##################
    if ml_type=="qsvc":
        ml, sampler, options = qpu_algorithm.qsvc_def(n_features, bcknd, folder)
    elif ml_type=="pegasos_qsvc":
        ml, sampler, options = qpu_algorithm.pegasos_def(n_features, bcknd, folder)
    
    print("Starting training...")
    start = time.time()
    ml.fit(train_features, train_labels)
    train_time = time.time() - start

    ################## results ##################
    print("\nSaving model...")
    ml.to_dill(f"{folder}/{filename}.model")

    results_save(ml_type, ml, sampler, bcknd, options, folder, train_time, n_features, d_file, filename, test_features)

def main():
    bcknd = "garnet"
    ml_type = "qsvc" # ["qsvc", "pegasos_qsvc"]
    d_n = 5
    d_size = 240

    d_file = f"kdd_3.14-scale_{d_n}-fpca_onehot-enc_{d_size}"

    training(ml_type, bcknd, d_file)
    
if __name__ == "__main__":
    main()

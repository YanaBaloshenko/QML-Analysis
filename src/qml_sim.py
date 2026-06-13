import numpy as np
import datetime
import time
import os
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

import algorithm

def report(folder, ml, ml_type, predictions):
    with open(f"{folder}/metadata.json", "r") as f:
        data = json.load(f)

    filename = data["filename"]

    d_file = data["d_file"]
    sets = np.load(f"dataset/{d_file}/{d_file}.npz")
    train_features = sets['train_features']
    train_labels = sets['train_labels']
    test_features = sets['test_features']
    test_labels = sets['test_labels']

    train_score = ml.score(train_features, train_labels)

    # report calculations
    if not predictions:
        test_score = 0.0
        report = "N/A: All QPU inference jobs failed or returned errors."
        cm_text = "N/A: No valid data to display."
        print("No valid predictions found. Skipping metrics calculation.")
    else:
        test_score = accuracy_score(test_labels, predictions)
        report = classification_report(test_labels, predictions, target_names=["Normal (0)", "Attack (1)"], zero_division=0.0)
        cm = confusion_matrix(test_labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        cm_text = f"""
                  Predicted Normal| Predicted Attack
Actual Normal (0): {tn:^16} | {fp:^16}
Actual Attack (1): {fn:^16} | {tp:^16}
"""
        
    # saving report
    with open(f"{folder}/{filename}_report.txt", "w") as f:
        f.write(f"Model {type(ml).__name__}\n")
        f.write("\n--- Dataset info ---\n")
        f.write(f"Data file used: {d_file}\n")
        f.write(f"Number of features: {data.get('n_features', 'N/A')}\n")
        f.write(f"Number of records: train - {train_features.shape[0]}, test - {test_features.shape[0]}\n")

        f.write("\n--- Training info ---\n")
        f.write(f"Number of classes: {data.get('num_classes', 'N/A')}\n")
        f.write(f"Training time: {data.get('train_time', 'N/A')} s\n")
        if train_score != None:
            f.write(f"Accuracy on training set: {train_score:.2f}\n")
        f.write(f"Accuracy: {test_score:.2f}\n")
        
        f.write("\n--- Confusion matrix (class 0 - normal, class 1 - attack) ---\n")
        f.write(f"{cm_text}\n")

        f.write("\n--- Classification Report ---\n")
        f.write(f"{report}\n")

        f.write("\n--- Model parameters ---\n")
        f.write(f"Number of qubits: {data.get('num_qubits', 'N/A')}\n")
        if ml_type =="vqc":
            f.write(f"Number of iterations: {data.get('nit', 'N/A')}\n") # czy optymalizator zatrzymał się bo "dotarł do celu" czy skończył mu się limit iteracji (maxiter from optimalizator)
            f.write(f"Number of Function Evaluations: {data.get('nfev', 'N/A')}\n") # ile razy optymalizator musiał uruchomić obwód kwantowy - płacić trzeba za każde uruchomienie (nfev), a nie za samą iterację
            f.write(f"Loss function type: {data.get('loss_name', 'N/A')}\n") # actual thing in fit_result
            f.write(f"Loss function value: {data.get('fun', 'N/A')}\n") # najniższa wartość funkcji straty - na jakim poziomie zatrzymał się trening
            f.write(f"Initial point (starting weights): {data.get('initial_point', 'N/A')}\n")
            f.write(f"Weights: {data.get('weights', 'N/A')}\n") # optimized weights after the training
            f.write(f"Final Gradient Magnitude: {data.get('jac', 'N/A')}\n") # if values are exactly zero everywhere right from the start - Barren Plateau
            f.write(f"Number of Jacobian Evaluations: {data.get('njev', 'N/A')}\n") # how many times the optimizer explicitly stopped to calculate that exact slope (the gradient) during the entire training process - might be very expensive (None for COBYLA)

            f.write("\n--- Neural network ---\n")
            f.write(f"Features sent to NN: {data.get('num_inputs', 'N/A')}\n")
            f.write(f"Number of weights (dimensionality): {data.get('num_weights', 'N/A')}\n") # dimension of a gradient in wchich the minimum of objective function is searched
            f.write(f"Final probability shape: {data.get('output_shape', 'N/A')}\n") # if it's binary or not (number of classes)

            f.write(f"\n--- Optimizer ({data.get('optimizer_name', 'N/A')}) set settings ---\n")
            for key, value in data.get("optimizer_settings", {}).items():
                f.write(f"{key}: {value}\n")
        else: # qsvc/pegasos specific metrics
            f.write(f"Number of Support Vectors: {data.get('support_vectors', 'N/A')}\n")
            options = data.get('options', None)
            if options != None:
                f.write("Model options:\n")
                for option in options:
                    f.write(f"{option}\n")

        if ml_type == "vqc":
            f.write("\n--- Sampler info ---\n")
            f.write(f"Default shots: {data['num_shots']}\n")
            if ml_type == "vqc" and data.get('num_shots') is not None:
                total_cost = data['nfev'] * data['num_shots']
                f.write(f"Total computational cost: {total_cost} shots\n")
            else:
                f.write(f"Total computational cost: Exact statevector calculation ({data['nfev']} circuit evaluations)\n") # for ideal StatevectorSampler

def results_save(ml_type, ml, o_list, folder, train_time, test_features, n_features, d_file, filename):
    print(f"\nSaving data to {folder}...")
    metadata = {
        "ml_type": ml_type,
        "train_time": train_time,
        "n_features": n_features,
        "d_file": d_file,
        "num_shots": None,
        "filename": filename,
        "bcknd": "ideal"
    }

    if ml_type=="vqc":
        metadata.update({
            "nit": int(ml.fit_result.nit) if ml.fit_result and ml.fit_result.nit is not None else None,
            "nfev": int(ml.fit_result.nfev) if ml.fit_result and ml.fit_result.nfev is not None else None,
            "fun": float(ml.fit_result.fun) if ml.fit_result and ml.fit_result.fun is not None else None,
            "jac": float(ml.fit_result.jac) if ml.fit_result and ml.fit_result.jac is not None else 0,
            "njev": int(ml.fit_result.njev) if ml.fit_result and ml.fit_result.njev is not None else 0,
            "initial_point": ml.initial_point.tolist() if ml.initial_point is not None else None,
            "optimizer_name": type(ml.optimizer).__name__,
            "optimizer_settings": {k: str(v) for k, v in ml.optimizer.settings.items()},
            "loss_name": type(ml.loss).__name__,
            "num_inputs": ml.neural_network.num_inputs,
            "num_weights": ml.neural_network.num_weights,
            "output_shape": ml.neural_network.output_shape[0],
            "num_qubits": ml.neural_network.num_qubits,
            "objective_func_vals": o_list,
            "weights": ml.weights.tolist() if ml.weights is not None else None
        })
    else: # qsvc/pegasos
        metadata.update({
            "num_qubits": ml.quantum_kernel.feature_map.num_qubits,
            "support_vectors": len(ml.support_vectors_) if hasattr(ml, 'support_vectors_') else 0,
            "options": o_list
        })

    print("Calculating predictions...")
    predictions = ml.predict(test_features)
    metadata.update({"predictions": predictions.tolist()})
    print("Predictions successfully saved.")

    with open(f"{folder}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    if ml_type=="vqc" and o_list: # objective function plot
        print("Generating plot...")
        plt.figure()
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.title("Objective function value against iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.plot(range(len(o_list)), o_list)
        plt.savefig(f"{folder}/{filename}_obj.png", bbox_inches="tight", dpi=300)

    plt.close('all') # RAM cleaning

    print("Saving report...")
    report(folder, ml, ml_type, predictions)

def training(ml_type, train_features, train_labels, test_features, n_features, d_file, folder, filename):
    if ml_type=="vqc":
        ml, _, _, _, o_list = algorithm.vqc_def(n_features, "ideal", None, folder, filename)
    elif ml_type=="qsvc":
        ml, _, _, _, o_list = algorithm.qsvc_def(n_features, "ideal", folder)
    elif ml_type=="pegasos_qsvc":
        ml, _, _, _, o_list = algorithm.pegasos_def(n_features, "ideal", folder)
    
    print("Starting training...")
    start = time.time()
    ml.fit(train_features, train_labels)
    train_time = time.time() - start

    results_save(ml_type, ml, o_list, folder, train_time, test_features, n_features, d_file, filename)

def prep(ml_type, d_file):
    d_path = f"dataset/{d_file}"
    date = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    r_folder = f"results/ideal/{ml_type}"
    filename = f"{ml_type}_data-{d_file}_time-{date}"
    folder = f"{r_folder}/{filename}"
    os.makedirs(folder, exist_ok=True)

    ################## data read ##################
    data = np.load(f"{d_path}/{d_file}.npz")
    train_features = data['train_features']
    train_labels = data['train_labels']
    test_features = data['test_features']
    n_features = train_features.shape[1]

    ################## training and saving results ##################
    training(ml_type, train_features, train_labels, test_features, n_features, d_file, folder, filename)

def main():
    ml_type = "pegasos_qsvc" # ["vqc", "qsvc", "pegasos_qsvc"]
    d_n = 6
    d_size = 260

    d_file = f"kdd_3.14-scale_{d_n}-fpca_onehot-enc_{d_size}"
    prep(ml_type, d_file)
    
if __name__ == "__main__":
    main()

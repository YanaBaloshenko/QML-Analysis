import os
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.gradients import ParamShiftSamplerGradient
from qiskit_machine_learning.algorithms.classifiers import VQC

def vqc_report(folder):
    # files prep
    with open(f"{folder}/metadata.json", "r") as f:
        data = json.load(f)

    filename = data["filename"]
    ml = VQC.from_dill(f"{folder}/{filename}.model")

    d_file = data["d_file"]
    sets = np.load(f"dataset/{d_file}/{d_file}.npz")
    
    train_features = sets['train_features']
    train_labels = sets['train_labels']
    test_features = sets['test_features']
    test_labels = sets['test_labels']

    num_rec = data["num_rec"]
    if num_rec is not None:
        train_features, train_labels, test_features, test_labels = train_features[:num_rec], train_labels[:num_rec], test_features[:num_rec], test_labels[:num_rec]
    
    predictions = data["predictions"]
    pca_weights = np.load(f"dataset/{d_file}/{d_file}_pcaweights.npy")

    sampler = StatevectorSampler()
    ml.neural_network.sampler = sampler
    ml.neural_network.gradient = ParamShiftSamplerGradient(sampler=sampler)

    # report calculations
    if not predictions or "ERROR_RETRIEVING_RESULTS" in predictions:
        train_score = 0.0
        test_score = 0.0
        report = "N/A: All QPU inference jobs failed or returned errors."
        cm_text = "N/A: No valid data to display."
        print("No valid predictions found. Skipping metrics calculation.")
    else:
        # calculating scores
        test_score = accuracy_score(test_labels, data["predictions"])
        train_score = ml.score(train_features, train_labels)

        # calculating cm and scikit report
        report = classification_report(test_labels, data["predictions"], target_names=["Normal (0)", "Attack (1)"])
        cm = confusion_matrix(test_labels, data["predictions"])
        tn, fp, fn, tp = cm.ravel()
        cm_text = f"""
                  Predicted Normal| Predicted Attack
Actual Normal (0): {tn:^16} | {fp:^16}
Actual Attack (1): {fn:^16} | {tp:^16}
"""
    
    # calculating feature importance
    sample_fi = test_features[0].reshape(1, -1)
    _ = ml.neural_network.forward(sample_fi, ml.weights)
    ml.neural_network.input_gradients = True
    in_grads, weight_grads = ml.neural_network.backward(sample_fi, ml.weights)
    
    encoded_names = np.load(f"dataset/{d_file}/{d_file}_names.npy", allow_pickle=True).tolist()
    fi_df = pd.DataFrame({
        'Original Feature': encoded_names,
        'Importance': np.abs(np.dot(in_grads[0][0], pca_weights).flatten())
    }).sort_values(by='Importance', ascending=False)

    fi = fi_df.to_string(
            index=False,                       # Hides the 0, 1, 2 row numbers
            justify='left',                    # Aligns the column headers nicely
            float_format=lambda x: f"{x:.6f}"  # Rounds the importance to 6 decimal places
        )

    # saving report
    with open(f"{folder}/{filename}_report.txt", "w") as f:
        f.write(f"Model {type(ml).__name__} | Job Id {data.get('job_id', 'N/A')}\n")
        f.write("\n--- Dataset info ---\n")
        f.write(f"Data file used: {d_file}\n")
        f.write(f"Number of features: {data["n_features"]}\n")
        f.write(f"Number of records: train - {train_features.shape[0]}, test - {test_features.shape[0]}\n")

        f.write("\n--- Training info ---\n")
        f.write(f"Number of classes: {data["num_classes"]}\n")
        f.write(f"Training time: {data.get('train_time', 'N/A')} s\n")
        f.write(f"Score on the training dataset: {train_score:.2f}\n")
        f.write(f"Score on the test dataset: {test_score:.2f}\n")
        
        f.write("\n--- Confusion matrix (class 0 - normal, class 1 - attack) ---\n")
        f.write(f"{cm_text}\n")

        f.write("\n--- Classification Report ---\n")
        f.write(f"{report}\n")

        f.write("\n--- Feature Importance ---\n")
        f.write(f"{fi}\n")

        f.write("\n--- Model parameters ---\n")
        f.write(f"Number of qubits: {data["num_qubits"]}\n")
        f.write(f"Number of iterations: {data["nit"]}\n") # czy optymalizator zatrzymał się bo "dotarł do celu" czy skończył mu się limit iteracji !!!!!!! maxiter from optimalizator
        f.write(f"Number of Function Evaluations: {data["nfev"]}\n") # ile razy optymalizator musiał uruchomić obwód kwantowy - płacić trzeba za każde uruchomienie (nfev), a nie za samą iterację
        f.write(f"Loss function type: {data["loss_name"]}\n") # actual thing in fit_result
        f.write(f"Loss function value: {data["fun"]}\n") # najniższa wartość funkcji straty - na jakim poziomie zatrzymał się trening
        f.write(f"Initial point (starting weights): {data["initial_point"]}\n")
        f.write(f"Weights: {data["weights"]}\n") # optimized weights after the training
        f.write(f"Final Gradient Magnitude: {data["jac"]}\n") # if values are exactly zero everywhere right from the start - Barren Plateau
        f.write(f"Number of Jacobian Evaluations: {data["njev"]}\n") # how many times the optimizer explicitly stopped to calculate that exact slope (the gradient) during the entire training process - might be very expensive (None for COBYLA)

        f.write("\n--- Neural network ---\n")
        f.write(f"Features sent to NN: {data["num_inputs"]}\n")
        f.write(f"Number of weights (dimensionality): {data["num_weights"]}\n") # imension of a gradient in wchich the minimum of objective function is searched
        f.write(f"Final probability shape: {data["output_shape"]}\n") # if it's binary or not (number of classes)

        f.write(f"\n--- Optimizer ({data['optimizer_name']}) settings ---\n")
        for key, value in data["optimizer_settings"].items():
            f.write(f"{key}: {value}\n")

        f.write("\n--- Sampler info ---\n")
        f.write(f"Default shots: {data["num_shots"]}\n")
        if data["num_shots"] is not None:
            total_cost = data["nfev"] * data["num_shots"]
            f.write(f"Total computational cost: {total_cost} shots\n")
        else:
            f.write(f"Total computational cost: Exact statevector calculation ({data['nfev']} circuit evaluations)\n") # for ideal StatevectorSampler


def vqr_report(folder):
    folder


def main():
    folder = "results/11052026_1149"
    vqc_report(folder)

if __name__ == "__main__":
    main()


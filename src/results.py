import json
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
 
from qiskit_machine_learning.algorithms import VQC, QSVC, PegasosQSVC

def report(folder):
    # files prep
    with open(f"{folder}/metadata.json", "r") as f:
        data = json.load(f)

    filename = data["filename"]
    ml_type = data.get("ml_type", filename.split('_')[0])

    if ml_type == "vqc":
        ml = VQC.from_dill(f"{folder}/{filename}.model")
    elif ml_type == "qsvc":
        ml = QSVC.from_dill(f"{folder}/{filename}.model")
    elif ml_type == "pegasos_qsvc":
        ml = PegasosQSVC.from_dill(f"{folder}/{filename}.model")
    else:
        raise ValueError("Unknown model type.")

    d_file = data["d_file"]
    sets = np.load(f"dataset/{d_file}/{d_file}.npz")

    train_features = sets['train_features']
    train_labels = sets['train_labels']
    test_features = sets['test_features']
    test_labels = sets['test_labels']

    predictions = data.get("predictions", [])
    if predictions and isinstance(predictions[0], list):
        predictions = [p[0] for p in predictions]
    if predictions and "ERROR" not in str(predictions[0]):
        predictions = [int(p) for p in predictions]

    # report calculations
    if not predictions or "ERROR_RETRIEVING_RESULTS" in predictions:
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

    if data['bcknd'] == "ideal":
        train_score = ml.score(train_features, train_labels)
    else:
        train_score = None
 
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

        f.write("\n--- Sampler info ---\n")
        f.write(f"Default shots: {data['num_shots']}\n")
        if ml_type == "vqc" and data.get('num_shots') is not None:
            total_cost = data['nfev'] * data['num_shots']
            f.write(f"Total computational cost: {total_cost} shots\n")

def main():
    folder = "results/ideal/vqc/vqc_data-kdd_3.14-scale_5-fpca_onehot-enc_240_backend-ideal_time-10062026_1215"
    report(folder)

if __name__ == "__main__":
    main()


import numpy as np
import datetime
import time
import os
import json
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from qiskit.visualization import (plot_circuit_layout, plot_error_map)
from qiskit import qpy

from qiskit_machine_learning.algorithms.classifiers import VQC

import algorithm
import results

def results_save(ml_type, ml, backend, sampler, bcknd, o_list, folder, train_time, n_features, d_file, filename, compiled_base_circuit, test_features):
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
        "bcknd": bcknd
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

    if bcknd == "ideal":
        print("Calculating predictions...")
        predictions = ml.predict(test_features)
        metadata.update({"predictions": predictions.tolist()})
        print("Predictions successfully saved.")

    with open(f"{folder}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Generating plots...")
    if compiled_base_circuit is not None:
        compiled_base_circuit.draw(output='mpl', idle_wires=False).savefig(f"{folder}/plots/{filename}_transpiled-circuit.png", dpi=300)
    else:
        print("No circuit, skipping transpilation plot...")
    if backend is not None and compiled_base_circuit is not None:
        plot_circuit_layout(compiled_base_circuit, backend).savefig(f"{folder}/plots/{filename}_hardware-layout.png", dpi=300)
        plot_error_map(backend).savefig(f"{folder}/plots/{filename}_error-map.png", dpi=300)
    else:
        print("No hardware backend, skipping layout plot...")
    
    # objective function plot
    if ml_type=="vqc" and o_list:
        plt.figure()
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.title("Objective function value against iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.plot(range(len(o_list)), o_list)
        plt.savefig(f"{folder}/{filename}_obj.png", bbox_inches="tight", dpi=300)

    plt.close('all') # RAM cleaning

    if bcknd == "ideal":
        results.report(folder)

def training(ml_type, bcknd, pretrained_weights, train_features, train_labels, test_features, n_features, d_file, folder, filename, pre_t):
    if ml_type=="vqc":
        ml, pm, sampler, backend, o_list = algorithm.vqc_def(n_features, bcknd, pretrained_weights, folder, filename)
    elif ml_type=="qsvc":
        ml, pm, sampler, backend, o_list = algorithm.qsvc_def(n_features, bcknd, folder)
    elif ml_type=="pegasos_qsvc":
        ml, pm, sampler, backend, o_list = algorithm.pegasos_def(n_features, bcknd, folder)
    
    start = time.time()
    ml.fit(train_features, train_labels)
    train_time = time.time() - start

    ################## results ##################
    print("\nSaving model...")
    ml.to_dill(f"{folder}/{filename}.model")

    if pre_t==True:
        pretrained_weights = ml.weights
        np.save(f"{folder}/pretrained_weights.npy", pretrained_weights)

    if ml_type == "vqc" and bcknd != "ideal":
        print("Preparing base circuit...")
        base_circuit = ml.neural_network.circuit
        compiled_base_circuit = pm.run(base_circuit)
        with open(f"{folder}/base-circuit.qpy", "wb") as f:
            qpy.dump(compiled_base_circuit, f)
    else:
        compiled_base_circuit = None

    results_save(ml_type, ml, backend, sampler, bcknd, o_list, folder, train_time, n_features, d_file, filename, compiled_base_circuit, test_features)

def prep(ml_type, bcknd, d_file):
    d_path = f"dataset/{d_file}"
    date = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    if bcknd == "ideal":
        r_folder = f"results/ideal/{ml_type}"
    else:
        r_folder = f"results/qpu/{ml_type}"
    filename = f"{ml_type}_data-{d_file}_backend-{bcknd}_time-{date}"
    folder = f"{r_folder}/{filename}"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"{folder}/plots", exist_ok=True)
    os.makedirs(f"{folder}/checkpoints", exist_ok=True)
    if bcknd != "ideal":
        pre_folder = f"{folder}/pretraining"
        os.makedirs(f"{folder}/pretraining", exist_ok=True)
        os.makedirs(f"{pre_folder}/plots", exist_ok=True)

    ################## data read ##################
    data = np.load(f"{d_path}/{d_file}.npz")
    train_features = data['train_features']
    train_labels = data['train_labels']
    test_features = data['test_features']
    n_features = train_features.shape[1]

    ################## training and saving results ##################
    if ml_type=="vqc":
        if (bcknd != "ideal"):
            print("Starting pre-training...")
            training(ml_type, "ideal", None, train_features, train_labels, test_features, n_features, d_file, pre_folder, filename, True)
            algorithm.objective_func_vals.clear() # clearing objective function values from pre-training
            pretrained_weights = np.load(f"{pre_folder}/pretrained_weights.npy")
        else:
            pretrained_weights = None
        print("Starting training...")
        training(ml_type, bcknd, pretrained_weights, train_features, train_labels, test_features, n_features, d_file, folder, filename, False)

    elif ml_type=="qsvc" or ml_type=="pegasos_qsvc":
        print("Starting training...")
        training(ml_type, bcknd, None, train_features, train_labels, test_features, n_features, d_file, folder, filename, False)
    else:
        raise ValueError("No such model defined.")  

def main():
    bcknd = "ideal"
    ml_type = "qsvc" # ["vqc", "qsvc", "pegasos_qsvc"]
    d_n = 5
    d_size = 240

    d_file = f"kdd_3.14-scale_{d_n}-fpca_onehot-enc_{d_size}"

    prep(ml_type, bcknd, d_file)
    
if __name__ == "__main__":
    main()

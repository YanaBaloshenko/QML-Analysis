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

import qpu_algorithm

def results_save(ml, backend, sampler, bcknd, objective_func_vals, folder, train_time, n_features, d_file, filename, compiled_base_circuit, test_features):
    num_shots = sampler.options.default_shots

    print(f"\nSaving data to {folder}...")
    metadata = {
        "ml_type": "vqc",
        "train_time": train_time,
        "n_features": n_features,
        "d_file": d_file,
        "num_shots": num_shots,
        "filename": filename,
        "bcknd": bcknd,
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
        "objective_func_vals": objective_func_vals,
        "weights": ml.weights.tolist() if ml.weights is not None else None
    }

    print("Calculating predictions...")
    predictions = ml.predict(test_features)
    metadata.update({"predictions": predictions.tolist()})
    print("Predictions successfully saved.")

    with open(f"{folder}/metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Generating plots...")
    if compiled_base_circuit is not None:
        compiled_base_circuit.draw(output='mpl', idle_wires=False).savefig(f"{folder}/{filename}_transpiled-circuit.png", dpi=300)
    else:
        print("No circuit, skipping transpilation plot...")
    if backend is not None and compiled_base_circuit is not None:
        plot_circuit_layout(compiled_base_circuit, backend).savefig(f"{folder}/{filename}_hardware-layout.png", dpi=300)
        plot_error_map(backend).savefig(f"{folder}/{filename}_error-map.png", dpi=300)
    else:
        print("No hardware backend, skipping layout plot...")
    
    if objective_func_vals: # objective function plot
        plt.figure()
        plt.rcParams["figure.figsize"] = (12, 6)
        plt.title("Objective function value against iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Objective function value")
        plt.plot(range(len(objective_func_vals)), objective_func_vals)
        plt.savefig(f"{folder}/{filename}_obj.png", bbox_inches="tight", dpi=300)

    plt.close('all') # RAM cleaning

def training(bcknd, train_features, train_labels, test_features, n_features, d_file, folder, filename, pre_t):
    ml, pm, sampler, backend, objective_func_vals = qpu_algorithm.vqc_def(n_features, bcknd, folder, filename)
    
    start = time.time()
    ml.fit(train_features, train_labels)
    train_time = time.time() - start

    ################## results ##################
    print("\nSaving model...")
    ml.to_dill(f"{folder}/{filename}.model")

    print("Preparing base circuit...")
    base_circuit = ml.neural_network.circuit
    compiled_base_circuit = pm.run(base_circuit)
    with open(f"{folder}/base-circuit.qpy", "wb") as f:
        qpy.dump(compiled_base_circuit, f)

    results_save(ml, backend, sampler, bcknd, objective_func_vals, folder, train_time, n_features, d_file, filename, compiled_base_circuit, test_features)

def prep(bcknd, d_file, pre_t):
    d_path = f"dataset/{d_file}"
    date = datetime.datetime.now().strftime("%d%m%Y_%H%M%S")
    r_folder = "results/qpu/vqc"
    filename = f"vqc_data-{d_file}_backend-{bcknd}_time-{date}"
    folder = f"{r_folder}/{filename}"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"{folder}/checkpoints", exist_ok=True)

    ################## data read ##################
    data = np.load(f"{d_path}/{d_file}.npz")
    train_features = data['train_features']
    train_labels = data['train_labels']
    test_features = data['test_features']
    n_features = train_features.shape[1]

    ################## training and saving results ##################
    print("Starting training...")
    training(bcknd, train_features, train_labels, test_features, n_features, d_file, folder, filename, False)
 

def main():
    bcknd = "ideal"
    d_n = 5
    d_size = 240
    d_file = f"kdd_3.14-scale_{d_n}-fpca_onehot-enc_{d_size}"

    prep(bcknd, d_file)
    
if __name__ == "__main__":
    main()

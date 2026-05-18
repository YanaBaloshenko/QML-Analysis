from ast import If

import numpy as np
import datetime
import time
import os
import json
from matplotlib import pyplot as plt
from qiskit.visualization import (plot_circuit_layout, plot_error_map)
from qiskit import qpy

from qiskit_machine_learning.algorithms.classifiers import VQC

import algorithm
import results
import batch

def results_save(ml_type, ml, backend, sampler, bcknd, objective_func_vals, folder, train_time, n_features, d_file, num_rec, filename, compiled_base_circuit):
    if bcknd != "ideal":
        num_shots = sampler.options.default_shots
    else:
        num_shots = None

    print("\nSaving data...")
    with open(f"{folder}/metadata.json", "w") as f:
        json.dump({
            "train_time": train_time,
            "n_features": n_features,
            "nit": int(ml.fit_result.nit),
            "nfev": int(ml.fit_result.nfev),
            "fun": float(ml.fit_result.fun),
            "jac": float(ml.fit_result.jac) if ml.fit_result.jac is not None else 0,
            "njev": int(ml.fit_result.njev) if ml.fit_result.njev is not None else 0,
            "initial_point": ml.initial_point.tolist() if ml.initial_point is not None else None,
            "optimizer_name": type(ml.optimizer).__name__,
            "optimizer_settings": {k: str(v) for k, v in ml.optimizer.settings.items()},
            "loss_name": type(ml.loss).__name__,
            "num_qubits": ml.num_qubits,
            "num_classes": ml.num_classes,
            "objective_func_vals": objective_func_vals,
            "weights": ml.weights.tolist(),
            "num_inputs": ml.neural_network.num_inputs,
            "num_weights": ml.neural_network.num_weights,
            "output_shape": ml.neural_network.output_shape[0],
            "d_file": d_file,
            "num_shots": num_shots,
            "filename": filename,
            "num_rec": num_rec,"bcknd": bcknd,
            "objective_func_vals": objective_func_vals
        }, f, indent=4)

    print("Generating plots...")
    if compiled_base_circuit is not None:
        compiled_base_circuit.draw(output='mpl', idle_wires=False).savefig(f"{folder}/plots/{filename}_transpiled-circuit.png", dpi=300)
    else:
        print("No circuit, skipping transpilation plot...")
    if backend is not None:
        plot_circuit_layout(compiled_base_circuit, backend).savefig(f"{folder}/plots/{filename}_hardware-layout.png", dpi=300)
        plot_error_map(backend).savefig(f"{folder}/plots/{filename}_error-map.png", dpi=300)
    else:
        print("No hardware backend, skipping layout plot...")
    
    # objective function plot
    plt.figure()
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.savefig(f"{folder}/plots/{filename}_obj.png", bbox_inches="tight", dpi=300)

    plt.close('all') # RAM cleaning

    if bcknd == "ideal":
        batch.batch_results(folder)
        if ml_type == "vqc":
            results.vqc_report(folder)

def vqc_training(bcknd, pretrained_weights, train_features, train_labels, n_features, num_rec, d_file, folder, filename, pre_t):
    ml, pm, sampler, backend, objective_func_vals = algorithm.vqc_def(n_features, bcknd, pretrained_weights, folder, filename)
    start = time.time()
    ml.fit(train_features, train_labels)
    train_time = time.time() - start

    ################## results ##################
    print("\nSaving model...")
    ml.to_dill(f"{folder}/{filename}.model")
    #ml = VQC.from_dill(f"{folder}/{filename}.model")

    if pre_t == True:
        pretrained_weights = ml.weights
        np.save(f"{folder}/pretrained_weights.npy", pretrained_weights)

    print("Preparing base circuit...")
    base_circuit = ml.circuit
    #base_circuit_meas = base_circuit.measure_all(inplace=False)
    compiled_base_circuit = pm.run(base_circuit)
    with open(f"{folder}/base-circuit.qpy", "wb") as f:
        qpy.dump(compiled_base_circuit, f)

    results_save("vqc", ml, backend, sampler, bcknd, objective_func_vals, folder, train_time, n_features, d_file, num_rec, filename, compiled_base_circuit)

def main():
    # preparing files
    ml_type = "vqc"
    pre_bcknd = "ideal"
    bcknd = "ideal"
    d_file = "kdd_3.14-scale_2-fpca_onehot-enc"
    date = datetime.datetime.now().strftime("%d%m%Y_%H%M")
    #date = "11052026_1149"
    folder = f"results/{date}"
    os.makedirs(folder, exist_ok=True)
    pre_folder = f"{folder}/pretraining"
    os.makedirs(f"{folder}/pretraining", exist_ok=True)
    os.makedirs(f"{folder}/plots", exist_ok=True)
    os.makedirs(f"{pre_folder}/plots", exist_ok=True)
    os.makedirs(f"{folder}/checkpoints", exist_ok=True)
    filename = f"{ml_type}_data-{d_file}_backend-{bcknd}_time-{date}"

    ################## data read ##################
    data = np.load(f"dataset/{d_file}/{d_file}.npz")
    train_features = data['train_features']
    test_features = data['test_features']
    train_labels = data['train_labels']
    test_labels = data['test_labels']
    n_features = train_features.shape[1]
    num_rec = 4
    train_features, train_labels = train_features[:num_rec], train_labels[:num_rec] # for faster testing, comment out for full dataset
    test_features, test_labels = test_features[:num_rec], test_labels[:num_rec]

    ################## training and saving results ##################
    if ml_type == "vqc":
        if pre_bcknd is not None:
            print("Starting pre-training...")
            vqc_training(pre_bcknd, None, train_features, train_labels, n_features, num_rec, d_file, pre_folder, filename, True)
            algorithm.objective_func_vals.clear() # clearing objective function values from pre-training
            pretrained_weights = np.load(f"{pre_folder}/pretrained_weights.npy")
        else:
            pretrained_weights = None
        print("Starting training...")
        vqc_training(pre_bcknd, pretrained_weights, train_features, train_labels, n_features, num_rec, d_file, folder, filename, False)
    elif ml_type=="vqr":
        ml, pm, sampler, backend, objective_func_vals = algorithm.vqr_def(n_features, bcknd)
    else:
        raise ValueError("No such model defined.")    
    
if __name__ == "__main__":
    main()

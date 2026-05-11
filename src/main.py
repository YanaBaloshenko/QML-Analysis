from ast import If

import numpy as np
import datetime
import time
import os
import json
from matplotlib import pyplot as plt
from qiskit.visualization import (plot_circuit_layout, plot_distribution, plot_histogram)

import algorithm as algorithm
import results as results

def main():
    # preparing files
    ml_type = "vqc"
    pre_bcknd = "ideal"
    bcknd = "ideal"
    d_file = "kdd_3.14-scale_2-fpca_onehot-enc"
    date = datetime.datetime.now().strftime("%d%m%Y_%H%M")
    folder = f"results/{date}"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"{folder}/plots", exist_ok=True)
    os.makedirs(f"{folder}/pretraining", exist_ok=True)
    os.makedirs(f"{folder}/checkpoints", exist_ok=True)
    filename = f"{ml_type}_data-{d_file}_backend-{bcknd}_time-{date}"

    ################## data read ##################
    data = np.load(f"dataset/{d_file}/{d_file}.npz")
    train_features = data['train_features']
    test_features = data['test_features']
    train_labels = data['train_labels']
    test_labels = data['test_labels']
    n_features = train_features.shape[1]
    train_features, train_labels = train_features[:4], train_labels[:4] # for faster testing, comment out for full dataset
    test_features, test_labels = test_features[:4], test_labels[:4]

    ################## training ##################
    if ml_type=="vqc":
        # pre-training for better initial point
        pre_ml, _, _, _, _ = algorithm.vqc_def(n_features, pre_bcknd, None, f"{folder}/pretraining", f"pre_{filename}")
        print("Starting pre-training...")
        pre_ml.fit(train_features, train_labels) # Let SPSA find a good local minimum
        pretrained_weights = pre_ml.weights
        np.save(f"{folder}/pretraining/pretrained_weights.npy", pretrained_weights)
        algorithm.objective_func_vals.clear() # clearing objective function values from pre-training
        
        ml, pm, sampler, backend, objective_func_vals = algorithm.vqc_def(n_features, bcknd, pretrained_weights, folder, filename) # qpu ml definition with initial point from pre-training

    elif ml_type=="vqr":
        ml, pm, sampler, backend, objective_func_vals = algorithm.vqr_def(n_features, bcknd)
    else:
        raise ValueError("No such model defined.")

    print("\nStarting training...")
    start = time.time()
    ml.fit(train_features, train_labels)
    train_time = time.time() - start

    ################## results ##################
    print("\nSaving results...")
    ml.to_dill(f"{folder}/{filename}.model")

    print("Preparing base circuit...")
    base_circuit = ml.circuit
    base_circuit_meas = base_circuit.measure_all(inplace=False)
    compiled_base_circuit = pm.run(base_circuit_meas)

    print("Preparing test circuits...")
    test_circuits = []
    for feature in test_features:
        state_circuit = compiled_base_circuit.assign_parameters({**dict(zip(ml.neural_network.input_params, feature)), **dict(zip(ml.neural_network.weight_params, ml.weights))})
        test_circuits.append(state_circuit)

    print("Geting data...")
    job = sampler.run(test_circuits)
    job_id = job.job_id()
    while job.status().name not in ['DONE', 'CANCELLED', 'ERROR']:
        print(f"[{time.strftime('%X')}] Job Status: {job.status().name}...")
        time.sleep(30)

    if job.status().name == 'DONE':
        print(f"[{time.strftime('%X')}] Job Status: {job.status().name}")
        batch_results = job.result()
    else:
        print(f"Job failed with status: {job.status().name}")

    print("Calculating predictions...")
    predictions = []
    for i in range(len(test_features)):
        counts = batch_results[i].data.meas.get_counts()
        top_bitstring = max(counts, key=counts.get)
        predictions.append(ml.neural_network.interpret(int(top_bitstring, 2)))

    print("Generating plots...")
    sampler_run = batch_results[0]
    bitstrings = sampler_run.data.meas.get_counts()
    total_shots = sum(bitstrings.values())
    dist = {state: count / total_shots for state, count in bitstrings.items()}

    # saving plots
    compiled_base_circuit.draw(output='mpl', idle_wires=False).savefig(f"{folder}/plots/{filename}_transpiled-circuit_.png", dpi=300)
    if backend is not None:
        plot_circuit_layout(compiled_base_circuit, backend).savefig(f"{folder}/plots/{filename}_hardware-layout.png", dpi=300)
    else:
        print("No hardware backend, skipping layout plot...")
    plot_histogram(bitstrings, title="Measurements - Histogram").savefig(f"{folder}/plots/{filename}_histogram.png", dpi=300)
    plot_distribution(dist, title="Quasi-probability").savefig(f"{folder}/plots/{filename}_distribution.png", dpi=300)

    # objective function plot
    plt.figure()
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.title("Objective function value against iteration")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.savefig(f"{folder}/plots/{filename}_obj.png", bbox_inches="tight", dpi=300)

    plt.close('all') # RAM cleaning

    if bcknd != "ideal":
        num_shots = sampler.options.default_shots
    else:
        num_shots = None

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
            "predictions": predictions,
            "filename": filename,
            "job_id": str(job_id)
        }, f, indent=4)

if __name__ == "__main__":
    main()

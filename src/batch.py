import numpy as np
import json
import time

from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.visualization import (plot_distribution, plot_histogram)
from qiskit import qpy

import algorithm

def batch_results(folder):
    with open(f"{folder}/metadata.json", "r") as f:
        data = json.load(f)

    filename = data["filename"]
    ml = VQC.from_dill(f"{folder}/{filename}.model")

    d_file = data["d_file"]
    sets = np.load(f"dataset/{d_file}/{d_file}.npz")
    num_rec = data["num_rec"]
    bcknd = data["bcknd"]

    test_features = sets['test_features']
    if num_rec is not None:
        test_features = sets['test_features'][:num_rec]
    
    with open(f"{folder}/base_circuit.qpy", "rb") as f:
        compiled_base_circuit = qpy.load(f)[0]

    sampler, _ = algorithm.backend_def(bcknd)

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
    
    batch_results = None # Initialize as None to prevent UnboundLocalError

    if job.status().name == 'DONE':
        print(f"[{time.strftime('%X')}] Job Status: {job.status().name}")
        batch_results = job.result()
    else:
        print(f"Job failed with status: {job.status()}")
        try:
            print("Attempting to force-retrieve results...")
            batch_results = job.result() 
            print("Recovery successful")
        except Exception as e:
            print(f"Recovery failed: {e}")
    
    print("Calculating predictions...")
    if batch_results is None:
        predictions = ["ERROR_RETRIEVING_RESULTS"] * len(test_features)
    else:
        predictions = []
        for i in range(len(test_features)):
            data_pr = batch_results[i].data
            counts = data_pr.meas.get_counts()
            top_bitstring = max(counts, key=counts.get)
            predictions.append(ml.neural_network.interpret(int(top_bitstring, 2)))

    data["predictions"] = predictions
    data["job_id"] = str(job_id)
    with open(f"{folder}/metadata.json", "w") as f:
        json.dump(data, f, indent=4)

    if batch_results is not None:
        print("Generating plot...")
        sampler_run = batch_results[0]
        bitstrings = sampler_run.data.meas.get_counts()
        total_shots = sum(bitstrings.values())
        threshold = 0.02

        dist = {state: count / total_shots for state, count in bitstrings.items()}
        filtered_dist = {state: prob for state, prob in dist.items() if prob > threshold}
        noise_mass = 1.0 - sum(filtered_dist.values())
        noise_text = f"Filtered Hardware Noise: {noise_mass:.1%}"

        plot_distribution(dist, title="Quasi-probability", legend=[noise_text]).savefig(f"{folder}/plots/{filename}_distribution.png", dpi=300)
    else:
        print("Skipping plot because QPU results could not be retrieved.")

def main():
    folder = ''
    batch_results(folder)

if __name__ == "__main__":
    main()

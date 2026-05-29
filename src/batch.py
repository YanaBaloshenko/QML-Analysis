import numpy as np
import json
import time

from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.algorithms import VQR
from qiskit.visualization import plot_distribution
from qiskit import transpile
from qiskit import qpy

import algorithm
import recovery

def batch_results(folder):
    with open(f"{folder}/metadata.json", "r") as f:
        data = json.load(f)

    filename = data["filename"]

    if "ml_type" in data:
        ml_type = data["ml_type"]
    else:
        ml_type = filename.split('_')[0]

    if ml_type=="vqc":
        ml = VQC.from_dill(f"{folder}/{filename}.model")
    if ml_type=="vqr":
        ml = VQR.from_dill(f"{folder}/{filename}.model")
    else:
        raise ValueError("Unknown model type.")

    d_file = data["d_file"]
    sets = np.load(f"dataset/{d_file}/{d_file}.npz")
    num_rec = data["num_rec"]
    train_bcknd = data["bcknd"]
    if "test_bcknd" in data:
        bcknd = data["test_bcknd"]
    else:
        bcknd = data["bcknd"]

    test_features = sets['test_features']
    if num_rec is not None:
        test_features = sets['test_features'][:num_rec]
    
    with open(f"{folder}/base-circuit.qpy", "rb") as f:
        compiled_base_circuit = qpy.load(f)[0]

    if ml_type=="vqc":
        primitive, _, backend = algorithm.backend_def(bcknd)
    elif ml_type=="vqr":
        _, primitive, backend = algorithm.backend_def(bcknd)

    if train_bcknd == "ideal" and backend is not None and hasattr(backend, 'target'):
        circuit = transpile(compiled_base_circuit, backend=backend, optimization_level=3)
    else:
        circuit = compiled_base_circuit

    print("Preparing test circuits...")
    test_circuits = []
    for feature in test_features:
        state_circuit = circuit.assign_parameters({**dict(zip(ml.neural_network.input_params, feature)), **dict(zip(ml.neural_network.weight_params, ml.weights))})
        test_circuits.append(state_circuit)

    print("Geting data...")
    if ml_type=="vqc":
        if backend is not None:
            num_shots = data.get("num_shots", 1024)
            job = backend.run(test_circuits, shots=num_shots) # 
        else:
            job = primitive.run(test_circuits)
    elif ml_type=="vqr":
        1
    job_id = job.job_id()

    while job.status().name not in ['DONE', 'CANCELLED', 'ERROR']:
        print(f"[{time.strftime('%X')}] Job Status: {job.status().name}...")
        time.sleep(30)

    batch_results = None

    if job.status().name == 'DONE':
        print(f"[{time.strftime('%X')}] Job Status: {job.status().name}")
        batch_results = job.result()
    else:
        print(f"Job failed with status: {job.status()}")

    print("Calculating predictions...")
    if batch_results is None:
        predictions = ["ERROR_RETRIEVING_RESULTS"] * len(test_features)
    else:
        predictions = []
        for i in range(len(test_features)):
            if backend is not None:
                counts = batch_results.get_counts(i)
            else:
                data_pr = batch_results[i].data
                counts = data_pr.meas.get_counts()
            
            top_bitstring = max(counts, key=counts.get)
            clean_bitstring = top_bitstring.replace(" ", "") # Good habit to prevent spacing errors
            predictions.append(ml.neural_network.interpret(int(clean_bitstring, 2)))

    data["predictions"] = predictions
    data["job_id"] = str(job_id)
    with open(f"{folder}/metadata.json", "w") as f:
        json.dump(data, f, indent=4)
    print("Predictions successfully saved to metadata.json")

    if batch_results is not None:
        print("Generating plot...")
        if backend is not None:
            bitstrings = batch_results.get_counts(0)
        else:
            sampler_run = batch_results[0]
            bitstrings = sampler_run.data.meas.get_counts()
            
        total_shots = sum(bitstrings.values())
        
    threshold = 0.02
    dist = {state: count / total_shots for state, count in bitstrings.items()} 
    filtered_dist = {state: prob for state, prob in dist.items() if prob > threshold}
    noise_mass = 1.0 - sum(filtered_dist.values())
    noise_text = f"Filtered Hardware Noise: {noise_mass:.1%}"

    plot_distribution(filtered_dist, title="Quasi-probability", legend=[noise_text]).savefig(f"{folder}/plots/{filename}_distribution.png", dpi=300)
    print("Plot generated.")

def main():
    folder = 'results/11052026_2134'
    batch_results(folder)

if __name__ == "__main__":
    main()

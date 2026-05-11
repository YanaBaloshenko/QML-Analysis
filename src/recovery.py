import numpy as np
import json

from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit.visualization import (plot_distribution, plot_histogram)
import algorithm

def recover_job(folder, target_id):
    with open(f"{folder}/metadata.json", "r") as f:
        data = json.load(f)

    filename = data["filename"]
    ml = VQC.from_dill(f"{folder}/{filename}.model")

    d_file = data["d_file"]
    sets = np.load(f"dataset/{d_file}/{d_file}.npz")
    num_rec = data["num_rec"]
    bcknd = data["bcknd"]

    test_features = sets['test_features'][:num_rec]
    _, backend = algorithm.backend_def(bcknd)

    # retrieve the job
    print(f"Retrieving Job ID: {target_id} from {bcknd}...")
    try:
        job = backend.retrieve_job(target_id)
    except Exception as e:
        print(f"Failed to find job on Resonance: {e}")
        return
    
    batch_results = job.result()

    print("Calculating predictions...")
    predictions = []
    for i in range(len(test_features)):
        counts = batch_results.get_counts(i)
        top_bitstring = max(counts, key=counts.get)
        clean_bitstring = top_bitstring.replace(" ", "")
        predictions.append(ml.neural_network.interpret(int(clean_bitstring, 2)))

    data["predictions"] = predictions
    data["job_id"] = str(target_id)
    with open(f"{folder}/metadata.json", "w") as f:
        json.dump(data, f, indent=4)
    print("Predictions successfully saved to metadata.json")

    print("Generating plot...")
    bitstrings = batch_results.get_counts(0)
    total_shots = sum(bitstrings.values())
    threshold = 0.02

    dist = {state: count / total_shots for state, count in bitstrings.items()} 
    filtered_dist = {state: prob for state, prob in dist.items() if prob > threshold}
    noise_mass = 1.0 - sum(filtered_dist.values())
    noise_text = f"Filtered Hardware Noise: {noise_mass:.1%}"

    plot_distribution(filtered_dist, title="Quasi-probability", legend=[noise_text]).savefig(f"{folder}/plots/{filename}_distribution.png", dpi=300)
    print("Plot generated. Recovery complete.")

def main():
    folder = "results/11052026_1149"
    id = "019e1675-7fc3-7321-b3b0-b8f319287379"
    recover_job(folder, id)

if __name__ == "__main__":
    main()
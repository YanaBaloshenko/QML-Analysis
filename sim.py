from iqm import qiskit_iqm
from iqm.qiskit_iqm import IQMFakeAphrodite
from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
from qiskit.primitives import BackendSamplerV2

from qiskit.circuit.library import zz_feature_map
from qiskit.circuit.library import real_amplitudes
#from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.optimizers import SPSA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_machine_learning.algorithms.classifiers import VQC

import numpy as np
import datetime
import time
from matplotlib import pyplot as plt


def vqc_def(n_features, callback_graph):

    ### initializing simulator ###
    aphrodite = IQMFakeAphrodite() # 54 qubits - not the same as on resonance
    garnet = IQMFakeGarnet() # 20 qubits
    backend = garnet

    sampler = BackendSamplerV2(backend=backend)

    ### defining qml ###
    print("\nFeature map...")
    feature_map = zz_feature_map(feature_dimension=n_features, reps=1)
    print("Ansatz...")
    ansatz = real_amplitudes(num_qubits=n_features, reps=3)
    print("Oprimizer...")
    optimizer = SPSA(maxiter=5)#, learning_rate=0.1, perturbation=0.2)

    # pass manager for transpilation; optimization_level=3 - qiskit będzie się bardzo starał maksymalnie uprościć obwód
    print("Transpilation...")
    pm = generate_preset_pass_manager(target=backend.target, optimization_level=3)

    print("Defining algorithm (VQC)...")
    vqc = VQC(
        sampler=sampler,
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=callback_graph,
        pass_manager=pm
    )

    return vqc

def main():
    data = np.load("data/kdd_3.14-scale_5-f.npz")
    train_features = data['train_features']
    test_features = data['test_features']
    train_labels = data['train_labels']
    test_labels = data['test_labels']
    n_features = train_features.shape[1]

    print(f"Number of features: {n_features}")
    print(f"Number of train records: {train_features.shape[0]}")
    print(f"Number of test records: {test_features.shape[0]}")
    print("\nStarting training...")

    objective_func_vals = []
    def callback_graph(weights, value):
        objective_func_vals.append(value)
        print(f"Iteracja: {len(objective_func_vals)} | Wartość funkcji celu: {value:.4f}")

    vqc = vqc_def(n_features, callback_graph)

    start = time.time()
    vqc.fit(train_features, train_labels)
    elapsed = time.time() - start

    print(f"\nTraining time: {round(elapsed)} seconds")

    # scores - make into normal raport
    train_score = vqc.score(train_features, train_labels)
    test_score = vqc.score(test_features, test_labels)
    print(f"Quantum VQC on the training dataset: {train_score:.2f}")
    print(f"Quantum VQC on the test dataset: {test_score:.2f}")

    date = datetime.datetime.now()
    with open(f"results/repots/f{n_features}_{date}.txt", "a") as f:
        f.write(f"Quantum VQC on the training dataset: {train_score:.2f}")
        f.write(f"Quantum VQC on the test dataset: {test_score:.2f}")

    # plot objective function - check out qiskit visualization toolkit
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.title("Objective function value against iteration (SPSA)")
    plt.xlabel("Iteration")
    plt.ylabel("Objective function value")
    plt.plot(range(len(objective_func_vals)), objective_func_vals)
    plt.show()

if __name__ == "__main__":
    main()

import os
import numpy as np
from dotenv import load_dotenv

from qiskit_machine_learning.algorithms.classifiers import VQC # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.algorithms.VQC.html
from qiskit_machine_learning.algorithms import QSVC # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.algorithms.QSVC.html
from qiskit_machine_learning.algorithms import PegasosQSVC # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.algorithms.PegasosQSVC.html
from qiskit_machine_learning.kernels import FidelityQuantumKernel, FidelityStatevectorKernel # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.kernels.FidelityQuantumKernel.html # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.kernels.FidelityStatevectorKernel.html
from qiskit_machine_learning.state_fidelities import ComputeUncompute # https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.state_fidelities.ComputeUncompute.html

from iqm.qiskit_iqm import IQMProvider
from qiskit.primitives import BackendSamplerV2
from qiskit.circuit.library import zz_feature_map # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit.circuit.library.ZZFeatureMap.html
from qiskit.circuit.library import real_amplitudes, efficient_su2 # https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RealAmplitudes # https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.EfficientSU2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager # https://quantum.cloud.ibm.com/docs/en/api/qiskit/transpiler

from qiskit_machine_learning.optimizers import COBYLA # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.optimizers.COBYLA.html
from qiskit_machine_learning.optimizers import SPSA # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.optimizers.SPSA.html

load_dotenv()
token = os.getenv("IQM_TOKEN")
iqm_link = "https://resonance.iqm.tech/"

objective_func_vals = []
def get_callback(folder, file_name):
    def callback(*args):
        if len(args) == 5: # SPSA signature: (nfev, weights, value, stepsize, accepted)
            weights = args[1]
            value = args[2]
        else: # default VQC signature: (weights, value)
            weights = args[0]
            value = args[1]

        objective_func_vals.append(value)
        iteration = len(objective_func_vals)
        print(f"Iteration: {iteration} | Objective function value: {value:.4f}")

        latest_path = (f"{folder}/checkpoints/{file_name}_latest_checkpoint.npy") # always save the absolute latest state (overwrites the previous one)
        np.save(latest_path, weights)
        if iteration % 5 == 0:  # save a permanent history file every 5 iterations
            history_path = (f"{folder}/checkpoints/{file_name}_checkpoint_iter_{iteration}.npy")
            np.save(history_path, weights)
            
    return callback

def backend_def(bcknd):
    backend = IQMProvider(iqm_link, quantum_computer=bcknd).get_backend() # wybranie qpu do połączenia
    sampler = BackendSamplerV2(backend=backend) # konfiguracja samplera
    sampler.options.default_shots = 512 # ilość obliczeń (shotów) jednego obwodu
    sampler.options.resilience_level = 1 # poziom mitygacji błędów:
                                         # 0 (bez mitygacji), 1 (bazowa readout error), 2 (metod quasi-probability)

    return sampler, backend

def vqc_def(n_features, bcknd, folder, file_name):
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"{folder}/checkpoints", exist_ok=True)

    sampler, backend = backend_def(bcknd)
    fm_reps = 1 # liczba powtórzeń dla mapy cech
    ansatz_reps = 1 # liczba powtórzeń dla ansatzu

    optimizer = SPSA(maxiter=10, learning_rate=0.02, perturbation=0.1) # konfiguracja optymalizatora
    pm = generate_preset_pass_manager(optimization_level=2, target=backend.target) # preset do transpilacji
                                                                                   # opt_lvl: 0 - bez optymalizacji, 1 - lekka, 2 - ciężka, 3 - największa)
    feature_map = zz_feature_map(feature_dimension=n_features, reps=fm_reps, entanglement='linear') # mapa cech z liniowym splątaniem
    ansatz = real_amplitudes(num_qubits=n_features, reps=ansatz_reps, entanglement='linear') # ansatz z liniowym splątaniem
    custom_callback = get_callback(folder, file_name) # funkcja zwrotna zapisująca wartości funkcji kosztu

    vqc = VQC( # definicja algorytmu
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=custom_callback,
        sampler=sampler,
        pass_manager=pm
    )

    return vqc, pm, sampler, backend, objective_func_vals

def qsvc_def(n_features, bcknd, folder):
    os.makedirs(folder, exist_ok=True)

    sampler, backend = backend_def(bcknd)
    fidelity = ComputeUncompute(sampler=sampler) # metoda wierności
    f_m = zz_feature_map(feature_dimension=n_features, reps=1, entanglement='linear')
    pm = generate_preset_pass_manager(optimization_level=0, target=backend.target)
    feature_map = pm.run(f_m.decompose()) # transpilacja mapy cech przed podaniem do kernela
    qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

    c = 0.5
    options = [f"C={c}", "class_weight='balanced'"] # lista podanych do modelu opcji

    qsvc = QSVC(quantum_kernel=qkernel, class_weight='balanced', C=c) # definicja modelu

    return qsvc, sampler, options

def pegasos_def(n_features, bcknd, folder):
    os.makedirs(folder, exist_ok=True)
    
    sampler, backend = backend_def(bcknd)
    fidelity = ComputeUncompute(sampler=sampler)
    f_m = zz_feature_map(feature_dimension=n_features, reps=1, entanglement='linear')
    pm = generate_preset_pass_manager(optimization_level=0, target=backend.target)
    feature_map = pm.run(f_m.decompose())
    qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

    c = 50
    ns = 750
    options = [f"C={c}", f"num_steps={ns}"] # lista wybranych wartości c i num_steps do raportu
    
    qsvc = PegasosQSVC(quantum_kernel=qkernel, C=c, num_steps=ns) # definicja modelu

    return qsvc, sampler, options


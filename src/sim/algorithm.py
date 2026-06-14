import os
import numpy as np
from dotenv import load_dotenv
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from qiskit_machine_learning.algorithms.classifiers import VQC # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.algorithms.VQC.html
from qiskit_machine_learning.algorithms import QSVC # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.algorithms.QSVC.html
from qiskit_machine_learning.algorithms import PegasosQSVC # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.algorithms.PegasosQSVC.html
from qiskit_machine_learning.kernels import FidelityQuantumKernel, FidelityStatevectorKernel # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.kernels.FidelityQuantumKernel.html # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.kernels.FidelityStatevectorKernel.html
from qiskit_machine_learning.state_fidelities import ComputeUncompute # https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.state_fidelities.ComputeUncompute.html

from iqm.qiskit_iqm import IQMProvider
from qiskit.primitives import StatevectorSampler, BackendSamplerV2
from qiskit.circuit.library import zz_feature_map # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit.circuit.library.ZZFeatureMap.html
from qiskit.circuit.library import real_amplitudes, efficient_su2 # https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.RealAmplitudes # https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.EfficientSU2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager # https://quantum.cloud.ibm.com/docs/en/api/qiskit/transpiler

from qiskit_machine_learning.optimizers import COBYLA # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.optimizers.COBYLA.html
from qiskit_machine_learning.optimizers import SPSA # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.optimizers.SPSA.html

load_dotenv()
token = os.getenv("IQM_TOKEN")
iqm_link = "https://resonance.iqm.tech/"

objective_func_vals = []
def get_callback():
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
            
    return callback

def vqc_def(n_features, folder):
    os.makedirs(folder, exist_ok=True)

    sampler = StatevectorSampler()
    fm_reps = 1 # liczba powtórzeń dla mapy cech
    ansatz_reps = 3 # liczba powtórzeń dla ansatzu

    optimizer = SPSA(maxiter=100) # konfiguracja optymalizatora
    feature_map = zz_feature_map(feature_dimension=n_features, reps=fm_reps) # mapa cech z ilością kubitów równej liczbie cech
    ansatz = efficient_su2(num_qubits=n_features, reps=ansatz_reps) # ansatz z ilością kubitów równej liczbie cech
    custom_callback = get_callback() # funkcja zwrotna zapisująca wartości funkcji kosztu

    vqc = VQC( # definicja algorytmu
        feature_map=feature_map,
        ansatz=ansatz,
        optimizer=optimizer,
        callback=custom_callback,
        sampler=sampler
    )

    return vqc, objective_func_vals

def qsvc_def(n_features, folder):
    os.makedirs(folder, exist_ok=True)
    
    feature_map = zz_feature_map(feature_dimension=n_features, reps=1)
    qkernel = FidelityStatevectorKernel(feature_map=feature_map) # symulowane jądro
    
    c = 0.5
    options = [f"C={c}", "class_weight='balanced'"] # lista podanych do modelu opcji

    qsvc = QSVC(quantum_kernel=qkernel, class_weight='balanced', C=c) # definicja algorytmu

    return qsvc, options

def pegasos_def(n_features, folder):
    os.makedirs(folder, exist_ok=True)
    
    feature_map = zz_feature_map(feature_dimension=n_features, reps=1)
    qkernel = FidelityStatevectorKernel(feature_map=feature_map)

    c = 50
    ns = 750
    options = [f"C={c}", f"num_steps={ns}"] # lista wybranych wartości c i num_steps do raportu
    
    qsvc = PegasosQSVC(quantum_kernel=qkernel, C=c, num_steps=ns) # definicja modelu

    return qsvc, options

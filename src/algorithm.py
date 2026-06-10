import os
import numpy as np
from dotenv import load_dotenv

from iqm import qiskit_iqm
from iqm.qiskit_iqm.fake_backends.fake_garnet import IQMFakeGarnet
from iqm.qiskit_iqm import IQMProvider
from qiskit.primitives import StatevectorSampler, BackendSamplerV2
from qiskit.primitives import StatevectorEstimator, BackendEstimatorV2
from qiskit.quantum_info import SparsePauliOp

from qiskit.circuit.library import zz_feature_map
from qiskit.circuit.library import real_amplitudes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_machine_learning.optimizers import COBYLA
from qiskit_machine_learning.optimizers import SPSA

from qiskit_machine_learning.algorithms.classifiers import VQC # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.algorithms.VQC.html
from qiskit_machine_learning.algorithms import VQR # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.algorithms.VQR.html
from qiskit_machine_learning.algorithms import QSVC # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.algorithms.QSVC.html
from qiskit_machine_learning.algorithms import QSVR # https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.algorithms.QSVR.html
from qiskit_machine_learning.kernels import FidelityQuantumKernel, FidelityStatevectorKernel
from qiskit_machine_learning.state_fidelities import ComputeUncompute

load_dotenv()
token = os.getenv("IQM_TOKEN")
iqm_link = "https://resonance.iqm.tech/"

spsa = SPSA(maxiter=20, learning_rate=0.02, perturbation=0.1) # SPSA doc https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.optimizers.SPSA.html

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
        
        if iteration % 5 == 0:  # save a permanent history file every 5 iterations (if QPU gets very noisy and ruins weights late in the run)
            history_path = (f"{folder}/checkpoints/{file_name}_checkpoint_iter_{iteration}.npy")
            np.save(history_path, weights)
            
    return callback

def backend_def(bcknd):
    if bcknd == "ideal":
        backend=None
        estimator = StatevectorEstimator()
        sampler = StatevectorSampler()
    elif bcknd == "f_garnet":
        backend=IQMFakeGarnet() # 20 qubits
        estimator = BackendEstimatorV2(backend=backend)
        sampler = BackendSamplerV2(backend=backend)
    elif bcknd == "sirius" or bcknd == "garnet" or bcknd == "emerald":
        backend = IQMProvider(iqm_link, quantum_computer=bcknd).get_backend() # docs https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.primitives.BackendSamplerV2
        
        estimator = BackendEstimatorV2(backend=backend) # !!!!!!!!!!!!!!!!!!!!!! check options
        estimator.options.default_precision = 0.05 # 0.05 is a sweet spot? - it's 400 shots per circuit (n = 1/precision^2)
        estimator.options.resilience_level = 1
        
        sampler = BackendSamplerV2(backend=backend)
        sampler.options.default_shots = 512
        sampler.options.resilience_level = 1 # resilience_level (int) – level of error mitigation to apply, valid values are 0 (no error mitigation), 1 (basic readout error mitigation), and 2 (advanced error mitigation using quasi-probability method)
    else:
        raise ValueError(f"Couldn't find backend: {bcknd}")
    return sampler, estimator, backend

def vqc_def(n_features, bcknd, initial_point, folder, file_name):
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"{folder}/plots", exist_ok=True)
    os.makedirs(f"{folder}/checkpoints", exist_ok=True)

    sampler, _, backend = backend_def(bcknd)
    feature_map = zz_feature_map(feature_dimension=n_features, reps=1, entanglement='linear') # feature map doc https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit.circuit.library.ZZFeatureMap.html
    ansatz = real_amplitudes(num_qubits=n_features, reps=1, entanglement='linear') # ansatz doc https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit.circuit.library.RealAmplitudes.html
    if hasattr(backend, 'target'):
        pm = generate_preset_pass_manager(optimization_level=2, target=backend.target) # transpilators docs https://quantum.cloud.ibm.com/docs/en/api/qiskit/transpiler
    else:
        pm = generate_preset_pass_manager(optimization_level=2) # for ideal backend, no need to specify target as it doesn't have any constraints on gates or connectivity

    if bcknd == "ideal":
        optimizer = SPSA(maxiter=50)
    else:
        optimizer = spsa

    custom_callback = get_callback(folder, file_name)

    print("Defining algorithm...")
    vqc = VQC(
        # num_qubits (also defined by feature map and ansatz)
        feature_map=feature_map,
        ansatz=ansatz,
        # loss (default - cross_entropy) - target loss function to be used in training
        optimizer=optimizer,
        #warm_start=True, # use weights from previous fit to start next fit
        initial_point=initial_point,
        callback=custom_callback,
        sampler=sampler,
        pass_manager=pm
        # interpret - callable that maps measured integer to another unsigned integer or tuple of unsigned integers (used as new indices for the (potentially sparse) output array, basic parity function used if None passed)
        # output_shape (default - 2) - for the underlying neural network generally equals to number of classes
    )

    # saving plots
    vqc.feature_map.draw(output='mpl').savefig(f"{folder}/plots/{file_name}_featuremap.png", dpi=300, bbox_inches='tight')
    vqc.ansatz.draw(output='mpl').savefig(f"{folder}/plots/{file_name}_ansatz.png", dpi=300, bbox_inches='tight')
    vqc.circuit.draw(output='mpl').savefig(f"{folder}/plots/{file_name}_complete-circuit.png", dpi=300, bbox_inches='tight')

    return vqc, pm, sampler, backend, objective_func_vals

def vqr_def(n_features, bcknd, initial_point, folder, file_name):
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"{folder}/plots", exist_ok=True)
    os.makedirs(f"{folder}/checkpoints", exist_ok=True)

    _, estimator, backend = backend_def(bcknd) 

    if bcknd == "ideal":
        optimizer = SPSA(maxiter=50)
    else:
        optimizer = spsa

    feature_map = zz_feature_map(feature_dimension=n_features, reps=1, entanglement='linear')
    ansatz = real_amplitudes(num_qubits=n_features, reps=1, entanglement='linear')
    
    observable = SparsePauliOp.from_list([("Z" * n_features, 1)]) # observable !!!!!!!!!!!!!!!!! check # the result would be in [-1, 1]

    if hasattr(backend, 'target'):
        pm = generate_preset_pass_manager(optimization_level=2, target=backend.target)
    else:
        pm = generate_preset_pass_manager(optimization_level=2)

    custom_callback = get_callback(folder, file_name)

    print("Defining algorithm...")
    vqr = VQR(
        #num_qubits (int | None) – for the underlying QNN, if None - derived from the feature map or ansatz
        feature_map=feature_map, # zz_feature_map() is default, for a single qubit regression problem - z_feature_map()
        ansatz=ansatz, # real_amplitudes() is default
        observable=observable, # (BaseOperator | None) observable to be measured in the underlying QNN
        #loss (str | Loss) – default is squared error
        optimizer=optimizer, # defaults to SLSQP
        #warm_start (bool) – use weights from previous fit to start next fit
        initial_point=initial_point,
        callback=custom_callback,
        estimator=estimator,
        pass_manager=pm
    )

    vqr.feature_map.draw(output='mpl').savefig(f"{folder}/plots/{file_name}_featuremap.png", dpi=300, bbox_inches='tight')
    vqr.ansatz.draw(output='mpl').savefig(f"{folder}/plots/{file_name}_ansatz.png", dpi=300, bbox_inches='tight')
    vqr.circuit.draw(output='mpl').savefig(f"{folder}/plots/{file_name}_complete-circuit.png", dpi=300, bbox_inches='tight')

    return vqr, pm, estimator, backend, objective_func_vals

def qsvc_def(n_features, bcknd, folder, file_name):
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"{folder}/plots", exist_ok=True)

    sampler, _, backend = backend_def(bcknd)
    f_m = zz_feature_map(feature_dimension=n_features, reps=1, entanglement='linear')

    if hasattr(backend, 'target'):
        pm = generate_preset_pass_manager(optimization_level=2, target=backend.target)
    else:
        pm = generate_preset_pass_manager(optimization_level=2)

    feature_map = pm.run(f_m)

    fidelity = ComputeUncompute(sampler=sampler)
    if bcknd == "ideal":
        qkernel = FidelityStatevectorKernel(feature_map=feature_map)
    else:
        qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

    print("Defining algorithm...")
    qsvc = QSVC(quantum_kernel=qkernel)

    qkernel.feature_map.draw(output='mpl').savefig(f"{folder}/plots/{file_name}_featuremap.png", dpi=300, bbox_inches='tight')

    return qsvc, pm, sampler, backend, [] # Empty list for objective func vals

def qsvr_def(n_features, bcknd, folder, file_name):
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"{folder}/plots", exist_ok=True)

    sampler, _, backend = backend_def(bcknd)
    f_m = zz_feature_map(feature_dimension=n_features, reps=1, entanglement='linear')

    if hasattr(backend, 'target'):
        pm = generate_preset_pass_manager(optimization_level=2, target=backend.target)
    else:
        pm = generate_preset_pass_manager(optimization_level=2)

    feature_map = pm.run(f_m)

    fidelity = ComputeUncompute(sampler=sampler)
    if bcknd == "ideal":
        qkernel = FidelityStatevectorKernel(feature_map=feature_map)
    else:
        qkernel = FidelityQuantumKernel(feature_map=feature_map, fidelity=fidelity)

    print("Defining algorithm...")
    qsvr = QSVR(quantum_kernel=qkernel)

    qkernel.feature_map.draw(output='mpl').savefig(f"{folder}/plots/{file_name}_featuremap.png", dpi=300, bbox_inches='tight')

    return qsvr, pm, sampler, backend, []
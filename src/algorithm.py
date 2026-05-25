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

load_dotenv()
token = os.getenv("IQM_TOKEN")
iqm_link = "https://resonance.iqm.tech/"

spsa = SPSA(maxiter=3, learning_rate=0.02, perturbation=0.05) # SPSA doc https://qiskit-community.github.io/qiskit-machine-learning/stubs/qiskit_machine_learning.optimizers.SPSA.html

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
        estimator.options.default_precision = 0.05 
        estimator.options.resilience_level = 1
        
        sampler = BackendSamplerV2(backend=backend)
        sampler.options.default_shots = 1024
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
    # pm:
    # optimization_level Level 0 disables all unnecessary optimizations; only transformations needed to make the circuit runnable are used.
    # Level 3 enables a full range of optimization techniques (can be very expensive in compilation time). It is not always guaranteed to produce the best results.
    # Qiskit defaults to optimization level 2, as a trade-off between compilation time and the expected amount of optimization.
    # If you need to ensure reproducibility of a compilation, pass a known integer to the seed_transpiler argument of the generator functions.
    # backend - (optional) can be used as the source of the default values for the basis_gates, coupling_map, and target. Any other arguments specified will take precedence
    # target - backend compilation target, coupling_map and basis_gates will be inferred from this argument if they are not set
    # basis_gates - list of basis gate names to unroll to (e.g: ['u1', 'u2', 'u3', 'cx'])
    # coupling_map - directed graph
    # dt - (float) backend sample time (resolution) in seconds, if None (default) and a backend is provided, backend.dt is used
    # initial_layout – initial position of virtual qubits on physical qubits
    # layout_method – pass for choosing initial qubit placement, valid choices are 'trivial', 'dense', and 'sabre'. This can also be the external plugin
    # routing_method – pass for routing qubits on the architecture, valid choices are 'basic', 'lookahead', 'sabre', and 'none' This can also be the external plugin
    # translation_method – method for translating gates to basis gates, valid choices 'translator', 'synthesis'. This can also be the external plugin
    # scheduling_method – pass for scheduling instructions, valid choices are 'alap' and 'asap'. This can also be the external plugin
    # approximation_degree – (float) heuristic dial used for circuit approximation (1.0=no approximation, 0.0=maximal approximation)
    # seed_transpiler – (int) sets random seed for the stochastic parts of the transpiler
    # unitary_synthesis_method – name of unitary synthesis method, by default 'default' is used
    # unitary_synthesis_plugin_config – (optional)(dict) only necessary when unitary synthesis plugin specified with unitary_synthesis_method argument
    # hls_config – (optional) configuration class passed directly to HighLevelSynthesis transformation pass, specifies high-level objects, the lists of synthesis algorithms and their parameters
    # init_method – plugin name for the init stage of the output
    # optimization_method – plugin name for the optimization stage of the output
    # qubits_initially_zero – (bool) indicates whether the input circuit is zero-initialized

    if bcknd == "ideal":
        optimizer = SPSA(maxiter=40)
    else:
        optimizer = spsa

    custom_callback = get_callback(folder, file_name)

    print("Defining algorithm (VQC)...")
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
        optimizer = SPSA(maxiter=40)
    else:
        optimizer = spsa

    feature_map = zz_feature_map(feature_dimension=n_features, reps=1, entanglement='linear')
    ansatz = real_amplitudes(num_qubits=n_features, reps=1, entanglement='linear')
    
    observable = SparsePauliOp.from_list([("Z" * n_features, 1)]) # observable !!!!!!!!!!!!!!!!! check

    if hasattr(backend, 'target'):
        pm = generate_preset_pass_manager(optimization_level=2, target=backend.target)
    else:
        pm = generate_preset_pass_manager(optimization_level=2)

    custom_callback = get_callback(folder, file_name)

    print("Defining algorithm (VQR)...")
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
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
import pandas as pd
import time
from matplotlib import pyplot as plt
from IPython.display import clear_output

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


### Initializing simulator ###
aphrodite = IQMFakeAphrodite() # 54 qubits - not the same as on resonance
garnet = IQMFakeGarnet() # 20 qubits
backend = garnet

sampler = BackendSamplerV2(backend=backend)

### dataset ###
df_train = pd.read_csv("data/nsl-kdd/KDDTrain+.txt") # reading files
df_test = pd.read_csv("data/nsl-kdd/KDDTest+.txt")

df_train = df_train.head(90).copy()
df_test = df_test.head(30).copy()

y_train = df_train['attack_type'].apply(lambda x: 0 if x == 'normal' else 1).values # changing to binary
y_test = df_test['attack_type'].apply(lambda x: 0 if x == 'normal' else 1).values

x_train_raw = df_train.drop(['attack_type', 'difficulty_level'], axis=1) # unnecessary columns drop
x_test_raw = df_test.drop(['attack_type', 'difficulty_level'], axis=1)

n_train_samples = len(x_train_raw)
x_combined = pd.concat([x_train_raw, x_test_raw], axis = 0)

x_combined_enc = pd.get_dummies(x_combined) # one-hot encoding !!!!!!!!!!!!!!!!!!!!! check

x_train_enc = x_combined_enc.iloc[:n_train_samples] # getting set back after encoding
x_test_enc = x_combined_enc.iloc[n_train_samples:]

### normalization ### !!!!!!! check ranges
scaler = MinMaxScaler(feature_range=(0, np.pi))
x_train_scaled = scaler.fit_transform(x_train_enc)
x_test_scaled = scaler.transform(x_test_enc)

### pca ###
n_features = 4
pca = PCA(n_components=n_features)
train_features = pca.fit_transform(x_train_scaled)
test_features = pca.transform(x_test_scaled)

train_labels = y_train # getting labels
test_labels = y_test

### defining qml ###
feature_map = zz_feature_map(feature_dimension=n_features, reps=1)
ansatz = real_amplitudes(num_qubits=n_features, reps=3)
optimizer = SPSA(maxiter=10, learning_rate=0.05, perturbation=0.1)

# pass manager for transpilation
# optimization_level=3 - qiskit będzie się bardzo starał maksymalnie uprościć obwód
pm = generate_preset_pass_manager(target=backend.target, optimization_level=3)

objective_func_vals = []
plt.rcParams["figure.figsize"] = (12, 6)

def callback_graph(n_evals, parameters, value, stepsize, accepted):
    objective_func_vals.append(value)
    print(f"Iteracja: {int(n_evals/3)} | Wartość funkcji celu: {value:.4f}")

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph,
    pass_manager=pm
)

# clear objective value history
objective_func_vals = []

start = time.time()
vqc.fit(train_features, train_labels)
elapsed = time.time() - start

print(f"Training time: {round(elapsed)} seconds")

train_score_q4 = vqc.score(train_features, train_labels)
test_score_q4 = vqc.score(test_features, test_labels)

print(f"Quantum VQC on the training dataset: {train_score_q4:.2f}")
print(f"Quantum VQC on the test dataset: {test_score_q4:.2f}")

plt.title("Objective function value against iteration (SPSA)")
plt.xlabel("Iteration")
plt.ylabel("Objective function value")
plt.plot(range(len(objective_func_vals)), objective_func_vals)
plt.show()

import json
import numpy as np

from qiskit_machine_learning.algorithms.classifiers import VQC

ml = VQC.from_dill("results/qpu/vqc/11052026_1149_qpu/vqc_data-kdd_3.14-scale_2-fpca_onehot-enc_backend-garnet_time-11052026_1149.model")
lr = ml.optimizer.learning_rate
perturb = ml.optimizer.perturbation
print("lr: ", lr)
print("perturb: ", perturb)
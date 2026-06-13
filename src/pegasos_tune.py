import numpy as np
import pandas as pd
import time
import os

from qiskit_machine_learning.algorithms import PegasosQSVC
from qiskit_machine_learning.kernels import FidelityStatevectorKernel
from qiskit.circuit.library import zz_feature_map
from sklearn.metrics import accuracy_score, f1_score

def hyperparameter_search(d_file):
    d_path = f"dataset/{d_file}"
    data = np.load(f"{d_path}/{d_file}.npz")
    
    train_features = data['train_features']
    train_labels = data['train_labels']
    test_features = data['test_features']
    test_labels = data['test_labels']
    n_features = train_features.shape[1]

    print("Kernel Matrix...)")
    start_kernel = time.time()
    feature_map = zz_feature_map(feature_dimension=n_features, reps=1)
    qkernel = FidelityStatevectorKernel(feature_map=feature_map)
    
    matrix_train = qkernel.evaluate(x_vec=train_features) # oblicznie macierzy treningowej
    matrix_test = qkernel.evaluate(x_vec=test_features, y_vec=train_features) # macierz testowa
    
    print(f"Finished. Time: {time.time() - start_kernel:.2f} s")

    print("\nGrid Search...")
    C_values = [100.0, 200.0, 300.0, 400.0, 500.0] # zbiór wartości parametru regularyzacji
    num_steps_values = [250, 500, 750, 1000, 1250, 1500, 1750] # zbiór wartości kroków algorytmu
    results_list = []
    
    for c in C_values:
        for steps in num_steps_values:
            ml = PegasosQSVC(C=c, num_steps=steps, precomputed=True, seed=42) # definicja modelu
            
            train_start = time.time()
            ml.fit(matrix_train, train_labels) # trenowanie
            train_time = time.time() - train_start
            predictions = ml.predict(matrix_test) # predykcje
            # skuteczność i f1-score z otrzymanych predykcji:
            acc = accuracy_score(test_labels, predictions)
            f1 = f1_score(test_labels, predictions, zero_division=0.0)
            
            results_list.append({ # zapisywanie wyników
                'C': c,
                'num_steps': steps,
                'Accuracy': acc,
                'F1_Score': f1,
                'Train_Time_s': train_time
            })

    results_df = pd.DataFrame(results_list)
    results_df = results_df.sort_values(by=['Accuracy', 'Train_Time_s'], ascending=[False, True]) # sortowanie po najlepszej dokładności, a potem po najkrótszym czasie treningu
    
    print("\nResults:")
    print(results_df.to_string(index=False))
    
    # Zapis do pliku CSV
    os.makedirs("results/ideal/pegasos_qsvc", exist_ok=True)
    csv_path = f"results/ideal/pegasos_qsvc/tuning_{d_file}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}.")

if __name__ == "__main__":
    dataset_file = "kdd_3.14-scale_6-fpca_onehot-enc_260" 
    hyperparameter_search(dataset_file)
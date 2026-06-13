import os
import time
import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.inspection import permutation_importance

import classic_plots

def calculate_feature_importance(model, test_features, test_labels, pca_weights, encoded_names):
    r = permutation_importance(model, test_features, test_labels, n_repeats=10, random_state=42)
    original_importances = np.abs(np.dot(r.importances_mean, pca_weights)) # Przemnożenie wyników PCA przez wagi, aby odzyskać wpływ na oryginalne cechy sieciowe
    fi_df = pd.DataFrame({
        'Original Feature': encoded_names,
        'Importance': original_importances
    }).sort_values(by='Importance', ascending=False)
    
    return fi_df

def train(ml_type, n_f, size):
    d_file = f"kdd_3.14-scale_{n_f}-fpca_onehot-enc_{size}"
    d_path = f"dataset/{d_file}"
    date = datetime.datetime.now().strftime("%d%m%Y_%H%M")
    f = f"results/classic/{ml_type}"
    os.makedirs(f, exist_ok=True)
    folder = f"{f}/{size}"
    os.makedirs(folder, exist_ok=True)
    filename = f"{ml_type}_data-{d_file}_time-{date}"
    save_path = f"{folder}/{filename}"
    os.makedirs(save_path, exist_ok=True)
    model_path = f"{save_path}/{filename}.joblib"
    file_path = f"{save_path}/{filename}_report.txt"
    
    ################## data read ##################
    data = np.load(f"{d_path}/{d_file}.npz")
    train_features = data['train_features']
    test_features = data['test_features']
    train_labels = data['train_labels']
    test_labels = data['test_labels']
    n_features = train_features.shape[1]
    
    pca_weights = np.load(f"{d_path}/{d_file}_pcaweights.npy")
    encoded_names = np.load(f"{d_path}/{d_file}_names.npy", allow_pickle=True).tolist()
    
    ################## training ##################
    if ml_type == "svc":
        ml = SVC(max_iter=10000, cache_size=2000, class_weight='balanced')
    elif ml_type == "pegasos":
        ml = SGDClassifier(loss='hinge', penalty='l2', max_iter=10000, random_state=42, early_stopping=True, n_iter_no_change=20, class_weight='balanced') # loss and penalty make it exactly pegasos algorithm
    else:
        raise ValueError("No such model defined.")
    
    print("Starting training...")
    start_train = time.time()
    ml.fit(train_features, train_labels)
    train_time = time.time() - start_train
    
    n_iter = getattr(ml, 'n_iter_', 'N/A') # odzyskiwanie liczby iteracji do zbieżności (zabezpieczenie przed listą w SVM)
    if isinstance(n_iter, np.ndarray):
        n_iter = n_iter[0] 
            
    start_inf = time.time()
    predictions = ml.predict(test_features)
    inference_time = time.time() - start_inf
        
    ################## saving results ##################
    print("Preparing results...")
    joblib.dump(ml, model_path)

    acc = accuracy_score(test_labels, predictions)
    report = classification_report(test_labels, predictions, target_names=["Normal (0)", "Attack (1)"], zero_division=0.0)
    cm = confusion_matrix(test_labels, predictions)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = "N/A", "N/A", "N/A", "N/A"
            
    cm_text = f"""                  Predicted Normal| Predicted Attack
Actual Normal (0): {tn:^16} | {fp:^16}
Actual Attack (1): {fn:^16} | {tp:^16}"""
    
    # model expressivity
    if ml_type=="svc":
        complexity_metric = "Number of Support Vectors"
        complexity_val = len(ml.support_)
    elif ml_type == "pegasos":
        complexity_metric = "Numbers of Weights & Biases"
        complexity_val = ml.coef_.size + ml.intercept_.size
    
    print(f"Saving report to {file_path}...")
    with open(file_path, "w") as f:
        f.write(f"Model {type(ml).__name__}\n")
        f.write("\n--- Dataset info ---\n")
        f.write(f"Data file used: {d_file}\n")
        f.write(f"Number of features: {n_features}\n")
        f.write(f"Number of records: train - {train_features.shape[0]}, test - {test_features.shape[0]}\n")

        f.write("\n--- Training info ---\n")
        f.write(f"Training time: {train_time} s\n")
        f.write(f"Inference time for {len(test_labels)} records: {inference_time:.4f} s\n")
        f.write(f"Number of iterations for convergance: {n_iter}\n")
        f.write(f"Models complexity ({complexity_metric}): {complexity_val}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        
        f.write("\n--- Confusion matrix (class 0 - normal, class 1 - attack) ---\n")
        f.write(f"{cm_text}\n")

        f.write("\n--- Classification Report ---\n")
        f.write(f"{report}\n")

        f.write("\n--- Model Architecture & Parameters ---\n")
        if ml_type=="svc":
            f.write(f"Kernel type: {ml.kernel}\n")
            f.write(f"Gamma parameter: {ml._gamma}\n")
            f.write(f"Number of Support Vectors: {len(ml.support_)}\n")
            f.write(f"Support Vector dimensions: {ml.support_vectors_.shape}\n")
        elif ml_type == "pegasos":
            f.write(f"Loss function: {ml.loss}\n")
            f.write(f"Penalty: {ml.penalty}\n")
            f.write(f"Number of trainable weights: {complexity_val}\n")

        f.write(f"\n--- Optimizer ({type(ml).__name__}) Settings ---\n")
        for key, value in ml.get_params().items():
            f.write(f"{key}: {value}\n")

    #classic_plots.collect_results(f"results/classic/{ml_type}")

def main():
    n_f = [5, 6, 7, 8, 9, 10, 11, 12]
    size = [100, 160, 200, 220, 240, 260, 280, 300]

    ml_type = ["svc", "pegasos"]

    #for ml in ml_type:
        #for n in n_f:
            #for s in size:
                #train(ml, n, s)

    for n in n_f:
        for s in size:
            train("pegasos", n, s)
    
    #train("svc", 5, 240)

if __name__ == "__main__":
    main()

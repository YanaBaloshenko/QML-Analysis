import numpy as np
import pandas as pd
import os
import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def prepare_data():
    range = np.pi # for normalization
    n_features = 2 # for pca
    folder = f"dataset/kdd_{range:.2f}-scale_{n_features}-fpca_onehot-enc"
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/kdd_{range:.2f}-scale_{n_features}-fpca_onehot-enc"

    print("Reading files...")
    df_train = pd.read_csv("dataset/nsl-kdd/KDDTrain+.txt") # reading files
    df_test = pd.read_csv("dataset/nsl-kdd/KDDTest+.txt")

    df_train['label'] = df_train['attack_type'].apply(lambda x: 0 if x == 'normal' else 1).values # changing to binary
    df_test['label'] = df_test['attack_type'].apply(lambda x: 0 if x == 'normal' else 1).values

    df_train = df_train.drop(['attack_type', 'difficulty_level'], axis=1) # unnecessary columns drop
    df_test = df_test.drop(['attack_type', 'difficulty_level'], axis=1)

    # Stratified Sampling (saving the proportions)
    print("Sampling...")
    _, df_train = train_test_split(df_train, test_size=125, stratify=df_train['label'], random_state=42) # in qiskit tutorial there are 150 samples so I'm using 150 samples split by around 85:15 proportions as in original dataset
    _, df_test = train_test_split(df_test, test_size=25, stratify=df_test['label'], random_state=42)

    y_train = df_train['label'].values
    y_test = df_test['label'].values

    x_train_raw = df_train.drop(['label'], axis=1) 
    x_test_raw = df_test.drop(['label'], axis=1)

    n_train_samples = len(x_train_raw)
    x_combined = pd.concat([x_train_raw, x_test_raw], axis = 0)

    print("One-Hot encoding...")
    x_combined_enc = pd.get_dummies(x_combined) # one-hot encoding !!!!!!!!!!!!!!!!!!!!! check
    encoded_feature_names = x_combined_enc.columns.tolist()
    np.save(f"{filename}_names.npy", encoded_feature_names)

    x_train_enc = x_combined_enc.iloc[:n_train_samples] # getting set back after encoding
    x_test_enc = x_combined_enc.iloc[n_train_samples:]

    ### normalization ### !!!!!!! check ranges
    print(f"Normalization for range (0, {range:.2f})...")
    scaler = MinMaxScaler(feature_range=(0, range))
    x_train_scaled = scaler.fit_transform(x_train_enc)
    x_test_scaled = scaler.transform(x_test_enc)

    ### pca ###
    print(f"PCA for {n_features} features...")
    pca = PCA(n_components=n_features)
    train_features = pca.fit_transform(x_train_scaled)
    test_features = pca.transform(x_test_scaled)

    train_labels = y_train # getting labels
    test_labels = y_test

    pca_weights = pca.components_
    np.save(f"{filename}_pcaweights.npy", pca_weights)

    # saving to .npz file
    np.savez(f"{filename}.npz",
             train_features=train_features, 
             test_features=test_features, 
             train_labels=train_labels, 
             test_labels=test_labels)
    
    print("Finished.")

if __name__ == "__main__":
    prepare_data()

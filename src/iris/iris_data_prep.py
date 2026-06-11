import numpy as np
import pandas as pd
import os
import math

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


def prepare_data(df_train, df_test, norm_range, n_features, filename):
    y_train = df_train['label'].values
    y_test = df_test['label'].values

    x_train_raw = df_train.drop(['label'], axis=1) 
    x_test_raw = df_test.drop(['label'], axis=1)

    print("One-Hot encoding...")
    x_train_enc = pd.get_dummies(x_train_raw, dtype=int)
    x_test_enc = pd.get_dummies(x_test_raw, dtype=int)
    x_train_enc, x_test_enc = x_train_enc.align(x_test_enc, join='left', axis=1, fill_value=0) # align test set to match train set columns exactly
    encoded_feature_names = x_train_enc.columns.tolist()
    np.save(f"{filename}_names.npy", encoded_feature_names)

    print(f"Normalization...")
    scaler = MinMaxScaler()
    x_train_scaled = scaler.fit_transform(x_train_enc)
    x_test_scaled = scaler.transform(x_test_enc)

    print(f"PCA for {n_features} features...")
    pca = PCA(n_components=n_features)
    train_features = pca.fit_transform(x_train_scaled)
    test_features = pca.transform(x_test_scaled)

    train_labels = y_train # getting labels
    test_labels = y_test

    pca_weights = pca.components_
    np.save(f"{filename}_pcaweights.npy", pca_weights)

    # saving to .npz file
    print(f"Saving to {filename}...")
    np.savez(f"{filename}.npz",
             train_features=train_features, 
             test_features=test_features, 
             train_labels=train_labels, 
             test_labels=test_labels)
    
    print("Finished.")

def main():
    norm_range = np.pi # for normalization
    n_features = 5 # for pca
    size = 240

    trn_size = math.ceil((size*85)/100) # 85:15 train to test
    tst_size = size-trn_size
    name = f"kdd_{norm_range:.2f}-scale_{n_features}-fpca_onehot-enc_{size}"
    f = f"dataset/{n_features}"
    os.makedirs(f, exist_ok=True)
    folder = f"{f}/{name}"
    os.makedirs(folder, exist_ok=True)
    filename = f"{folder}/{name}"

    print("Reading files...")
    df_train = pd.read_csv("dataset/nsl-kdd/KDDTrain+.txt") # reading files
    df_test = pd.read_csv("dataset/nsl-kdd/KDDTest+.txt")

    df_train['label'] = df_train['attack_type'].apply(lambda x: 0 if x == 'normal' else 1).values # changing to binary
    df_test['label'] = df_test['attack_type'].apply(lambda x: 0 if x == 'normal' else 1).values

    df_train = df_train.drop(['attack_type', 'difficulty_level'], axis=1) # unnecessary columns drop
    df_test = df_test.drop(['attack_type', 'difficulty_level'], axis=1)

    print("Sampling...") # stratified sampling - that's to reduce the dataset while saving the proportions
    _, df_train = train_test_split(df_train, test_size=trn_size, stratify=df_train['label'], random_state=42)
    _, df_test = train_test_split(df_test, test_size=tst_size, stratify=df_test['label'], random_state=42)

    prepare_data(df_train, df_test, norm_range, n_features, filename)

if __name__ == "__main__":
    main()

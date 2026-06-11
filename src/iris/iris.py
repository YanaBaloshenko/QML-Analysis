import numpy as np
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from qiskit_machine_learning.utils import algorithm_globals
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

import qml
import iris.iris_data_prep as iris_data_prep

def main():
    norm_range = np.pi # for normalization
    n_features = 2 # for pca
    size = 150

    name = f"iris_{n_features}-fpca_onehot-enc_{size}"
    d_folder = f"dataset/{name}"
    os.makedirs(d_folder, exist_ok=True)
    d_filename = f"{d_folder}/{name}"

    iris_data = load_iris()
    #features = iris_data.data
    #labels = iris_data.target
    #features = MinMaxScaler().fit_transform(features)
    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    algorithm_globals.random_seed = 123
    #train_features, test_features, train_labels, test_labels = train_test_split(features, labels, train_size=0.8, random_state=algorithm_globals.random_seed)
    df['label'] = iris_data.target
    df_train, df_test = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=algorithm_globals.random_seed)

    iris_data_prep.prepare_data(df_train, df_test, norm_range, n_features, d_filename)

if __name__ == "__main__":
    main()
import json
import numpy as np

with open("results/11052026_2134/metadata.json", "r") as f:
    data = json.load(f)

sets = np.load(f"dataset/kdd_3.14-scale_2-fpca_onehot-enc/kdd_3.14-scale_2-fpca_onehot-enc.npz")
print(len(sets["test_features"]))
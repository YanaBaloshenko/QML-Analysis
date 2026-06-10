import os
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

def parse_result_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    record_match = re.search(r"Number of records: train - (\d+), test - (\d+)", content)
    feat_match = re.search(r"Number of features: (\d+)", content)
    acc_match = re.search(r"Accuracy: ([\d\.]+)", content)
    
    if not all([record_match, feat_match, acc_match]):
        return None
    
    train_size = int(record_match.group(1))
    test_size = int(record_match.group(2))
    total_size = train_size + test_size
    features = int(feat_match.group(1))
    accuracy = float(acc_match.group(1))
    
    return {
        'dataset_size': total_size,
        'features': features,
        'accuracy': accuracy
    }

def collect_all_results(root_folder):
    data = []
    for root, dirs, files in os.walk(root_folder): # os.walk travels through all subdirectories
        for file in files:
            if file.endswith("_report.txt"):
                file_path = os.path.join(root, file)
                parsed = parse_result_file(file_path)

                if parsed:
                    data.append(parsed)
    return pd.DataFrame(data)

def plot(df, ml):
    heatmap_data = df.pivot_table(index='features', columns='dataset_size', values='accuracy')
    plt.figure(figsize=(10, 6))
    sns.heatmap(heatmap_data, annot=True, cmap='viridis')
    plt.title('Accuracy Heatmap (Features vs Dataset Size)')
    plt.savefig(f"results/classic/plots/{ml}_heatmap.png", bbox_inches="tight", dpi=300)
    plt.close('all')

def main():
    ml = 'svr' # ['svc', 'svr', 'mlpr']
    df = collect_all_results(f'results/classic/{ml}')

    plot(df, ml)

if __name__ == "__main__":
    main()
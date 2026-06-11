import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def qsvc_plots(results_dict):
    sorted_items = sorted(results_dict.items()) # Sorts the dictionary by keys (C values) in ascending order
    c_values, accuracies = zip(*sorted_items) # Unpack the sorted items into two separate lists
    c_labels = [str(c) for c in c_values]

    plt.figure(figsize=(8, 5))
    plt.plot(c_labels, accuracies, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)
    plt.xlabel('Regularization Parameter (C)', fontsize=12, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    plt.title('QSVC Performance based on Regularization parameter', fontsize=14)
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig("results/ideal/plots/qsvc_accuracy_vs_c_dict.png", dpi=300)
    plt.close()

def vqc_plots():
    runs = ['Run 1', 'Run 2', 'Run 3', 'Run 4', 'Run 5']
    times = [324.62, 322.18, 324.9, 324.06, 327.74]
    accuracies = [0.53, 0.67, 0.75, 0.81, 0.61]
    combined_labels = [f"{runs[i]}\n({times[i]}s)" for i in range(len(runs))] # combine the run name and time into a multi-line label

    plt.figure(figsize=(8, 5))
    plt.plot(combined_labels, accuracies, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)
    unique_accuracies = sorted(list(set(accuracies)))
    plt.yticks(unique_accuracies)
    plt.xlabel('Execution Instance & Training Time', fontsize=12, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    plt.title('VQR: Test Accuracy per Execution for RealAmplitudes Ansatz', fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(min(accuracies) - 0.02, max(accuracies) + 0.02)
    plt.tight_layout()
    plt.savefig("results/ideal/vqc/plots/vqc_accuracy-time_spsa.png", dpi=300)
    plt.close()

def main():
    qsvc_dict = {
        0.1: 0.69,
        0.3: 0.69,
        0.4: 0.75,
        0.5: 0.81,
        0.6: 0.78,
        0.7: 0.75,
        0.8: 0.75,
        0.9: 0.69,
        1.0: 0.69,
        10.0: 0.67,
        100.0: 0.67
    } # format: {C_value: accuracy}
    #qsvc_plots(qsvc_dict)

    vqc_plots()
    

if __name__ == "__main__":
    main()
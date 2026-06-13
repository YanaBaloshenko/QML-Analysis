import pandas as pd
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
    #times = [324.62, 322.18, 324.9, 324.06, 327.74] # spsa
    #accuracies = [0.53, 0.67, 0.75, 0.81, 0.61]
    times = [94.25, 94.98, 95.23, 95.79, 93.83] # cobyla
    accuracies = [0.67, 0.58, 0.67, 0.47, 0.61]
    #times = [121.47, 126.29, 124.45, 125.41, 125.73] # realamplitudes
    #accuracies = [0.50, 0.61, 0.64, 0.69, 0.61]
    #times = [141.6, 143.11, 144.1, 148.88, 144.96] # efficientsu2
    #accuracies = [0.58, 0.53, 0.58, 0.64, 0.72]
    combined_labels = [f"{runs[i]}\n({times[i]}s)" for i in range(len(runs))] # combine the run name and time into a multi-line label

    plt.figure(figsize=(8, 5))
    plt.plot(combined_labels, accuracies, marker='o', linestyle='-', color='#1f77b4', linewidth=2, markersize=8)
    unique_accuracies = sorted(list(set(accuracies)))
    plt.yticks(unique_accuracies)
    plt.xlabel('Execution Instance & Training Time', fontsize=12, fontweight='bold')
    plt.ylabel('Test Accuracy', fontsize=12, fontweight='bold')
    plt.title('VQC: Test Accuracy per Execution for COBYLA Optimizer', fontsize=14)
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.ylim(min(accuracies) - 0.02, max(accuracies) + 0.02)
    plt.tight_layout()
    plt.savefig("results/ideal/vqc/plots/vqc_accuracy-time_cobyla.png", dpi=300)
    plt.close()

def pegasos_plots():
    data = {
        'C': [100, 400, 500, 100, 100],
        'num_steps': [250, 500, 500, 750, 1750],
        'Accuracy': [0.65, 0.72, 0.75, 0.68, 0.70] # <-- WSTAW TUTAJ SWOJE WYNIKI
    }
    df = pd.DataFrame(data)

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")

    # 3. Rysowanie bąbelków
    scatter = plt.scatter(
        x=df['C'], 
        y=df['num_steps'], 
        s=df['Accuracy'] * 1000,  # Rozmiar kropki (przeskalowany do widoczności)
        c=df['Accuracy'],         # Kolor na podstawie wyniku
        cmap='viridis',           # Mapa kolorów (od ciemnofioletowego do żółtego)
        alpha=0.8,
        edgecolors="black",
        linewidth=1.5
    )

    # 4. Dodanie dokładnych wartości liczbowych nad kropkami
    for i in range(len(df)):
        plt.text(df['C'].iloc[i], df['num_steps'].iloc[i] + 40, 
              f"{df['Accuracy'].iloc[i]:.3f}", 
              horizontalalignment='center', size='medium', color='black', weight='bold')

    # 5. Etykiety i kosmetyka
    cbar = plt.colorbar(scatter)
    cbar.set_label('Accuracy', rotation=270, labelpad=15, weight='bold')

    plt.title('Wpływ hiperparametrów na skuteczność modelu (Pegasos QSVC)', fontsize=14, pad=20, weight='bold')
    plt.xlabel('Parametr regularyzacji (C)', fontsize=12, weight='bold')
    plt.ylabel('Liczba iteracji (num_steps)', fontsize=12, weight='bold')

    # Wymuszenie szerszych marginesów, żeby kropki nie ucinały się na krawędziach
    plt.xlim(0, 600)
    plt.ylim(0, 2000)

    plt.tight_layout()
    plt.savefig("pegasos_bubble_plot.png", dpi=300)
    plt.show()
    plt.close('all')

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

    pegasos_plots()
    

if __name__ == "__main__":
    main()
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. Define the files we renamed and their labels
data_map = {
    'm-PPO (Proposed)': '../results/mPPO_final.mat',
    'Standard MSE': '../results/MSE_final.mat',
    'Standard MAE': '../results/MAE_final.mat',
    'ITAE': '../results/ITAE_final.mat',
    'Sparse': '../results/Sparse_final.mat'
}

def plot_final_comparison():
    plt.figure(figsize=(12, 7))
    
    # Standard colors for high-quality research papers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, (label, path) in enumerate(data_map.items()):
        if os.path.exists(path):
            data = sio.loadmat(path)
            # 'y_rl' contains the process output over time
            y = data['y_rl'].flatten()
            plt.plot(y, label=label, color=colors[i], linewidth=2)
        else:
            print(f"⚠️ Warning: {path} not found. Skipping {label}.")

    # 2. Add the Setpoint (Target) line
    # For CS1, the setpoint is usually 1.0
    plt.axhline(y=1.0, color='black', linestyle='--', label='Setpoint', alpha=0.8, linewidth=1.5)

    # 3. Final Formatting for LNMIIT Presentation
    plt.title("Comparative Step Response: Unstable Process $G(s) = \\frac{1}{s-1}$", fontsize=16)
    plt.xlabel("Sampling Instants ($k$)", fontsize=14)
    plt.ylabel("System Output $y(k)$", fontsize=14)
    plt.legend(loc='lower right', fontsize=12)
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.ylim([0, 1.5]) # Adjust based on your overshoot
    
    # Save the high-resolution version for your report
    plt.tight_layout()
    plt.savefig("../results/BTP_Comparison_Overlay.png", dpi=300)
    print("✅ Success! Your overlay graph is saved as: results/BTP_Comparison_Overlay.png")
    plt.xlim([0, 100])   # Zoom into the first 100 steps
    plt.ylim([0.8, 1.2])    # Focus closely on the setpoint (1.0)

    # Save the "Zoomed View" (shows m-PPO precision)
    plt.savefig("../results/BTP_Zoomed_Comparison.png", dpi=300)
    print("✅ Zoomed graph saved.")

if __name__ == "__main__":
    plot_final_comparison()
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os

# --- ACADEMIC STYLING ---
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.4,
    'grid.linestyle': '--',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 10
})

# --- CONFIGURATION ---
folder = "matlab results"
dt = 0.1
setpoint = 1.0

# Mapping labels to their refreshed .mat files
files = {
    'PPO':              'ppo_matlab_matrix.mat',
    'SAC':              'sac_matlab_matrix.mat',
    'DDPG':             'ddpg_matlab_matrix.mat',
    'A2C':              'a2c_matlab_matrix.mat'
}

# Colors matching your previous training curves for consistency
colors = {
    'm-PPO (Proposed)': '#e74c3c', # Red
    'PPO':              '#3498db', # Blue
    'SAC':              '#2ecc71', # Green
    'DDPG':             '#9b59b6', # Purple
    'A2C':              '#f39c12'  # Orange
}

plt.figure(figsize=(12, 7))

# 1. Plot the Setpoint Line
plt.axhline(y=setpoint, color='black', linestyle='-', linewidth=1.5, label='Setpoint ($1.0$)', alpha=0.8)

print("--- GENERATING COMBINED RESPONSE PLOT ---")

for label, filename in files.items():
    path = os.path.join(folder, filename)
    
    if os.path.exists(path):
        data = sio.loadmat(path)
        y = data['y_rl'].flatten()
        t = np.arange(len(y)) * dt
        
        # Limit plot time to 15 seconds for clarity
        limit = int(15 / dt) 
        plt.plot(t[:limit], y[:limit], color=colors[label], linewidth=2, label=label)
        print(f"Added {label} to plot.")
    else:
        print(f"Skipping {label}: File not found.")

# --- GRAPH ANNOTATIONS ---
plt.title("Figure 4.6: Combined Step Response Comparison", fontweight='bold', pad=20)
plt.xlabel("Time (seconds)", fontweight='bold')
plt.ylabel("System Output ($y$)", fontweight='bold')
plt.legend(loc='upper right', frameon=True, shadow=True)

# Focus the Y-axis on the interesting behavior (0 to 2.5 usually captures all overshoots)
plt.ylim(0, 2.5) 
plt.xlim(0, 15)

plt.tight_layout()
plt.savefig("Figure_4_6_Combined_Response.png", dpi=300)
print("\nSUCCESS: Figure_4_6_Combined_Response.png generated.")
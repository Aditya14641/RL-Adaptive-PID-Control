import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import os

# --- SETTINGS ---
FOLDER_PATH = "matlab results"
SETPOINT = 1.0
DT = 0.1

# Mapping filenames to display labels
ALGO_FILES = {
    'PPO':              'ppo_matlab_matrix.mat',
    'SAC':              'sac_matlab_matrix.mat',
    'DDPG':             'ddpg_matlab_matrix.mat',
    'A2C':              'a2c_matlab_matrix.mat'
}

def calculate_metrics(y, t, sp):
    # 1. Rise Time (Time to first hit 100% of Setpoint)
    cross_idx = np.where(y >= sp)[0]
    rt = t[cross_idx[0]] if len(cross_idx) > 0 else t[-1]
    
    # 2. Peak Overshoot (%)
    peak_val = np.max(y)
    ov = max(0.0, ((peak_val - sp) / sp) * 100)
    
    return rt, ov

# --- DATA EXTRACTION ---
names = []
rise_times = []
overshoots = []

for label, filename in ALGO_FILES.items():
    path = os.path.join(FOLDER_PATH, filename)
    if os.path.exists(path):
        data = sio.loadmat(path)
        y = data['y_rl'].flatten()
        t = np.arange(len(y)) * DT
        rt, ov = calculate_metrics(y, t, SETPOINT)
        
        names.append(label)
        rise_times.append(rt)
        overshoots.append(ov)

# --- HELPER FOR SORTING ---
def get_sorted_data(names, values):
    # Sorts the data by value (descending/ascending) as per guidelines
    combined = sorted(zip(names, values), key=lambda x: x[1])
    return zip(*combined)

# --- FIGURE 4.7: RISE TIME BAR GRAPH ---
plt.figure(figsize=(10, 6))
s_names_rt, s_values_rt = get_sorted_data(names, rise_times)
colors_rt = ['#e74c3c' if 'Proposed' in n else '#3498db' for n in s_names_rt]

bars = plt.bar(s_names_rt, s_values_rt, color=colors_rt, edgecolor='black', alpha=0.8)
plt.title("Figure 4.7: Rise Time Comparison", fontweight='bold', fontsize=14)
plt.ylabel("Rise Time (seconds)", fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}s', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("Figure_4_7_RiseTime.png", dpi=300)
print("Generated Figure 4.7: Rise Time Comparison")

# --- FIGURE 4.8: OVERSHOOT BAR GRAPH ---
plt.figure(figsize=(10, 6))
s_names_ov, s_values_ov = get_sorted_data(names, overshoots)
colors_ov = ['#e74c3c' if 'Proposed' in n else '#95a5a6' for n in s_names_ov]

bars = plt.bar(s_names_ov, s_values_ov, color=colors_ov, edgecolor='black', alpha=0.8)
plt.title("Figure 4.8: Peak Overshoot Comparison", fontweight='bold', fontsize=14)
plt.ylabel("Overshoot (%)", fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig("Figure_4_8_Overshoot.png", dpi=300)
print("Generated Figure 4.8: Overshoot Comparison")
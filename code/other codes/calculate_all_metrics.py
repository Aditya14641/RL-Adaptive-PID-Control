import scipy.io as sio
import numpy as np
import os

# --- SETTINGS ---
FOLDER_PATH = "matlab results"
SETPOINT = 1.0
DT = 0.1  # Sampling time in seconds
OUTPUT_FILE = "Performance_Metrics_Table.txt"

# Map the display names to your specific filenames
FILES_TO_PROCESS = {
    'PPO':              'ppo_matlab_matrix.mat',
    'SAC':              'sac_matlab_matrix.mat',
    'DDPG':             'ddpg_matlab_matrix.mat',
    'A2C':              'a2c_matlab_matrix.mat'
}

def calculate_metrics(y, t, sp):
    error = sp - y
    
    # 1. Rise Time (Time to first hit 100% of Setpoint)
    cross_idx = np.where(y >= sp)[0]
    rise_time = t[cross_idx[0]] if len(cross_idx) > 0 else t[-1]
    
    # 2. Settling Time (Time to stay within 2% band of Setpoint)
    within_2_percent = np.abs(error) <= 0.02 * sp
    settling_time = 0
    for i in range(len(y)-1, -1, -1):
        if not within_2_percent[i]:
            settling_time = t[i+1] if i+1 < len(t) else t[-1]
            break
            
    # 3. Peak Overshoot (%)
    peak_val = np.max(y)
    overshoot = max(0.0, ((peak_val - sp) / sp) * 100)
    
    # 4. Integral Square Error (ISE)
    ise = np.sum(error**2) * DT
    
    # 5. Integral Absolute Error (IAE)
    iae = np.sum(np.abs(error)) * DT
    
    # 6. Steady-State Error (SSE)
    # Using the absolute average error of the last 10 samples
    sse = np.abs(np.mean(error[-10:]))
    
    return [rise_time, settling_time, overshoot, ise, iae, sse]

# --- MAIN EXECUTION ---
results = []
header = ["Algorithm", "Rise Time (s)", "Settling Time (s)", "Overshoot (%)", "ISE", "IAE", "SSE"]

print(f"Analyzing files in {FOLDER_PATH}...")

for display_name, filename in FILES_TO_PROCESS.items():
    path = os.path.join(FOLDER_PATH, filename)
    
    if os.path.exists(path):
        try:
            data = sio.loadmat(path)
            y = data['y_rl'].flatten()
            t = np.arange(len(y)) * DT
            
            metrics = calculate_metrics(y, t, SETPOINT)
            results.append([display_name] + metrics)
            print(f"Successfully processed: {display_name}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
    else:
        print(f"Warning: {filename} not found. Skipping...")

# --- GENERATE TXT FILE ---
with open(OUTPUT_FILE, "w") as f:
    # Write Table Header
    f.write("="*95 + "\n")
    f.write(f"{header[0]:<20} | {header[1]:<10} | {header[2]:<10} | {header[3]:<10} | {header[4]:<8} | {header[5]:<8} | {header[6]:<8}\n")
    f.write("-" * 95 + "\n")
    
    # Write Data Rows
    for row in results:
        f.write(f"{row[0]:<20} | {row[1]:<10.3f} | {row[2]:<10.3f} | {row[3]:<10.2f} | {row[4]:<8.3f} | {row[5]:<8.3f} | {row[6]:<8.5f}\n")
    
    f.write("="*95 + "\n")
    f.write(f"Metrics generated on: {DT}s sample time | Setpoint: {SETPOINT}\n")

print(f"\nSUCCESS: The comparison table has been saved to {OUTPUT_FILE}")
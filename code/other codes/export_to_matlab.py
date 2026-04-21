import scipy.io as sio
import os
import numpy as np

# 1. Define the source files
data_map = {
    'mPPO': '../results/mPPO_final.mat',
    'MSE': '../results/MSE_final.mat',
    'MAE': '../results/MAE_final.mat',
    'ITAE': '../results/ITAE_final.mat',
    'Sparse': '../results/Sparse_final.mat'
}

master_data = {}

for label, path in data_map.items():
    if os.path.exists(path):
        print(f"Reading {label}...")
        raw_data = sio.loadmat(path)
        
        # Check if process output exists
        if 'y_rl' in raw_data:
            y_values = raw_data['y_rl']
            master_data[f'y_{label}'] = y_values
            
            # Also grab control effort if available
            if 'u_rl' in raw_data:
                master_data[f'u_{label}'] = raw_data['u_rl']
            
            # Generate Time and Setpoint vectors only once
            if 't' not in master_data:
                # If 't' isn't in the file, we create it (0, 1, 2, ... N)
                num_samples = y_values.shape[1] if len(y_values.shape) > 1 else len(y_values)
                master_data['t'] = np.arange(num_samples)
                
                # Create a setpoint vector of 1.0 (standard for your Case Study 1)
                master_data['setpoint'] = np.ones(num_samples)
        else:
            print(f"⚠️ Warning: 'y_rl' not found in {path}")
    else:
        print(f"⚠️ Warning: File not found: {path}")

# 2. Save the master file
if master_data:
    sio.savemat('../results/Master_BTP_Results.mat', master_data)
    print("\n🚀 Success! Master_BTP_Results.mat created in the results folder.")
else:
    print("\n❌ Error: No data was found. Check your file paths!")
import scipy.io as sio
import numpy as np
import os

# Define the file paths for the results we renamed
data_map = {
    'm-PPO (Proposed)': '../results/mPPO_final.mat',
    'Standard MSE': '../results/MSE_final.mat',
    'Standard MAE': '../results/MAE_final.mat',
    'ITAE': '../results/ITAE_final.mat',
    'Sparse': '../results/Sparse_final.mat'
}

def generate_performance_table():
    table_lines = []
    
    # Define table headers
    header = f"{'Control Strategy':<20} | {'ISE':<12} | {'IAE':<12} | {'Steady-State Error':<20}"
    separator = "-" * len(header)
    
    table_lines.append(header)
    table_lines.append(separator)
    
    print("Gathering data and calculating metrics...")
    
    for label, path in data_map.items():
        if os.path.exists(path):
            # Load MATLAB file
            data = sio.loadmat(path)
            
            # y_rl is the process output; flatten it to a 1D array
            y = data['y_rl'].flatten()
            setpoint = 1.0  # Target value for Case Study 1
            
            # Calculate the error vector
            error = setpoint - y
            
            # Calculate metrics
            ise = np.sum(np.square(error))
            iae = np.sum(np.abs(error))
            sse = np.abs(error[-1]) # Error at the final sampling instant
            
            # Format the row for the table
            row = f"{label:<20} | {ise:<12.4f} | {iae:<12.4f} | {sse:<20.6f}"
            table_lines.append(row)
            print(f"✅ Calculated metrics for: {label}")
        else:
            print(f"⚠️ Skipping {label}: File not found at {path}")

    # Define the output path in the results folder
    output_file = '../results/Performance_Comparison_Table.txt'
    
    # Save the table as a text file
    with open(output_file, 'w') as f:
        f.write("\n".join(table_lines))
    
    print(f"\n🚀 Success! The summary table is saved at: {output_file}")

if __name__ == "__main__":
    generate_performance_table()
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# --- BULLETPROOF ACADEMIC STYLING (No Seaborn Required) ---
# This section completely replaces the plt.style.use() command to prevent all OSErrors
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.7
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.color'] = '#b0b0b0'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 14, 'axes.titlesize': 16})

# --- YOUR EXACT FILE PATHS ---
log_files = {
    'm-PPO': 'logs/CS1/CS1_SystemSimplePID_PPO_AR_True_2_use_sde_False_ES_True_extra_BetterES_SystemFix/0.monitor.csv', 
    'SAC': 'logs/CS1/CS1_SystemSimplePID_SAC_AR_True_2_use_sde_False_ES_True_extra_BetterES_SystemFix/0.monitor.csv',   
    'DDPG': 'logs/CS1/CS1_SystemSimplePID_DDPG_AR_True_2_use_sde_False_ES_True_extra_BetterES_SystemFix/0.monitor.csv',  
    'A2C': 'logs/CS1/CS1_SystemSimplePID_A2C_AR_True_2_use_sde_False_ES_True_extra_BetterES_SystemFix/0.monitor.csv'     
}

colors = {'m-PPO': '#e74c3c', 'SAC': '#2ecc71', 'DDPG': '#9b59b6', 'A2C': '#f39c12'}
smoothing_window = 50 

plt.figure(figsize=(10, 6))

for algo, file_path in log_files.items():
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, skiprows=1)
        episodes = np.arange(len(df))
        rewards = df['r'].values
        smoothed_rewards = pd.Series(rewards).rolling(window=smoothing_window, min_periods=1).mean()
        
        # 1. Plot Individual Graphs
        fig_ind, ax_ind = plt.subplots(figsize=(8, 5))
        ax_ind.plot(episodes, rewards, alpha=0.3, color=colors[algo], label='Raw Reward')
        ax_ind.plot(episodes, smoothed_rewards, color=colors[algo], linewidth=2, label='Smoothed Reward')
        ax_ind.set_title(f'Training Episode vs Reward ({algo})', fontweight='bold')
        ax_ind.set_xlabel('Episodes')
        ax_ind.set_ylabel('Reward')
        ax_ind.legend()
        plt.tight_layout()
        
        # Save individual graph safely
        safe_filename = f'{algo.replace("-", "")}_reward_plot.png'
        fig_ind.savefig(safe_filename, dpi=300)
        plt.close(fig_ind)
        print(f"Saved individual plot: {safe_filename}")

        # 2. Add to Combined Graph
        plt.plot(episodes, smoothed_rewards, label=algo, color=colors[algo], linewidth=2)
    else:
        print(f"WARNING: Could not find {file_path}. Please check the path!")

# Formatting the Combined Graph
plt.title('Figure 4.13: Combined Training Reward Comparison', fontweight='bold', pad=15)
plt.xlabel('Episodes')
plt.ylabel('Smoothed Reward')
plt.legend(loc='best')
plt.tight_layout()

plt.savefig('combined_reward_plot.png', dpi=300)
print("Saved combined plot: combined_reward_plot.png")

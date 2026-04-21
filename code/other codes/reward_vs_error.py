import numpy as np
import matplotlib.pyplot as plt

# --- ACADEMIC STYLING ---
plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16
})

# --- DATA GENERATION ---
# Simulating a range of Error (e = Setpoint - Process Variable)
error = np.linspace(-2.0, 2.0, 500)

# Common Reward Functions used in RL-PID
reward_linear = -np.abs(error)              # Negative Absolute Error
reward_quadratic = -(error**2)             # Negative Squared Error (Standard)
reward_m_ppo = -np.log(np.abs(error) + 1.0) # Logarithmic (Smoother gradients)

# --- PLOTTING ---
plt.figure(figsize=(10, 6))

plt.plot(error, reward_quadratic, color='#e74c3c', linewidth=3, label='Quadratic Reward (Standard)')
plt.plot(error, reward_linear, color='#3498db', linestyle='--', linewidth=2, label='Linear Reward')
plt.plot(error, reward_m_ppo, color='#2ecc71', linestyle='-.', linewidth=2, label='m-PPO Proposed Reward')

# --- ANNOTATIONS ---
plt.axvline(x=0, color='black', linewidth=1.2) # Setpoint line
plt.axhline(y=0, color='black', linewidth=1.2) # Max reward line

plt.annotate('Maximum Reward\n(System at Setpoint)', 
             xy=(0, 0), xytext=(0.5, -0.5),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             fontsize=10, fontweight='bold')

plt.title("Figure 4.15: Reward Function vs. System Error", fontweight='bold', pad=15)
plt.xlabel("Error ($e = r - y$)", fontweight='bold')
plt.ylabel("Reward Signal ($R$)", fontweight='bold')
plt.legend(loc='lower center', shadow=True)

# Shading the "Stability Zone"
plt.fill_between(error, -0.1, 0, where=(np.abs(error) < 0.1), 
                 color='green', alpha=0.2, label='Stability Zone')

plt.tight_layout()
plt.savefig("Figure_4_15_Reward_vs_Error.png", dpi=300)
print("SUCCESS: Figure 4.15 generated.")
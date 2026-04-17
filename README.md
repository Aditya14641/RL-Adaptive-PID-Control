# RL-Adaptive PID Control for Unstable Processes

This repository contains the implementation of a **Modified Proximal Policy Optimization (m-PPO)** reinforcement learning architecture designed to act as a high-level supervisor for a traditional PID controller. Unlike standard PID tuning, this approach is completely data-driven and does not require a mathematical model of the process.

## 🚀 Project Overview
Control of unstable processes is challenging due to their dynamic nature and stability issues. This project implements the **m-PPO** algorithm to adaptively tune $K_p$, $\tau_i$, and $\tau_d$ gains in real-time to ensure desired servo and regulatory performance.

### Key Features
* **m-PPO Algorithm:** A robust policy gradient method that tackles issues like catastrophic performance drops during training.
* **Modified Reward Function:** A combination of clipped Integral Squared Error (ISE) and a constant positive bonus $r_{add}$ when the error is low.
* **Early Stopping Criterion:** A safety mechanism that terminates training if the process variable crosses a threshold, preventing the agent from learning from unbounded output data.
* **Action Repeat:** Improves credit assignment by applying an action for a user-defined number of steps, which is essential for handling processes with slow dynamics.

## 🛠️ Control Architecture
The RL agent operates in a **supervisory hierarchy** above the conventional regulatory PID controller.

* **State:** A combination of the current setpoint $y_{sp}(t)$ and the current and previous process variables $y_o(t), y_o(t-1)$.
* **Action:** The agent predicts optimal PID parameters ($K_p$, $\tau_i$, $\tau_d$).
* **Reward:** Uses a modified reward $r_t = r_{ise} + r_{add}$ to handle the unbounded nature of unstable processes.

## 📊 Case Studies
The proposed RL-PID controller is validated on various processes:
* **Linear Unstable Systems:** First-order systems ($G(s) = \frac{1}{s-1}$) and second-order systems with two unstable poles.
* **Nonlinear Unstable Systems:** Jacketed Continuous Stirred Tank Reactor (CSTR) performing an exothermic reaction.
* **MIMO Complex Systems:** Attitude control (roll, pitch, yaw) of an Unmanned Aerial Vehicle (UAV).

## 📈 Performance Analysis
Comparison with classical PID, DDPG-PID, and A2C-PID shows that the m-PPO PID controller exhibits faster settling times and significantly less overshoot. It demonstrates high tolerance to measurement noise and effective generalization to untrained setpoints.

---
**Author:** Aadit  
**University:** The LNM Institute of Information Technology (LNMIIT)

## 📚 References
> **T. Shuprajhaa, S. K. Sujit, and K. Srinivasan**, "Reinforcement learning based adaptive PID controller design for control of linear/nonlinear unstable processes," *Applied Soft Computing*, vol. 128, p. 109450, 2022.

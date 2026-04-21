from pathlib import Path
import gym
import matplotlib.pyplot as plt
import numpy as np
from control.matlab import c2d, ssdata, tf, tf2ss
from gym import spaces
from pid_controller import PID

from test_model import evaluate
from utils import fig2data

uinit = 0.0
umin = -100.0
umax = 100
yinit = 0.0
delt = 0.1
slew_rate = 20.0
num = [1]
dem = [1, -1]
ksp = 10
disturbance = False
deterministic = False
disturbance_value = 0.2


class System:
    def __init__(
        self,
        uinit=uinit,
        yinit=yinit,
        num=num,
        dem=dem,
        delt=delt,
        ttfinal=None,
        disturbance=disturbance,
        deterministic=deterministic,
        disturbance_value=disturbance_value,
    ):
        # Simulation settings
        self.delt = delt  # sample time
        self.ttfinal = ttfinal  # final simulation time
        self.ksp = ksp
        self.slew_rate = slew_rate
        self.umin = umin
        self.umax = umax
        self.input_low = np.array([self.umin])
        self.input_high = np.array([self.umax])
        self.disturbance_value = disturbance_value

        self.uinit = uinit
        self.yinit = yinit
        self.num = num
        self.dem = dem
        self.disturbance = disturbance
        self.deterministic = deterministic

        self.linear_tf = tf(self.num, self.dem)
        self.linear_ss = tf2ss(self.linear_tf)
        #  discretize the plant with a sample time, delt
        self.sysd_plant = c2d(self.linear_ss, self.delt)
        [self.phi, self.gamma, self.cd, self.dd] = ssdata(self.sysd_plant)
        self.xinit = np.zeros((self.sysd_plant.A.shape[0], 1))

        self.reset()

    @property
    def state_names(self):
        names = ["Setpoint(k)", "Output(k)", "Output(k-1)"]
        assert len(names) == self.n_states
        return names

    @property
    def input_names(self):
        names = ["Input(k)"]
        assert len(names) == self.n_actions
        return names

    @property
    def n_states(self):
        return len(self.get_state())

    @property
    def n_actions(self):
        return self.input_low.shape[0]

    def reset(self):
        del_sp = 200
        sp_min = 0
        sp_max = 2
        if self.deterministic:
            self.r = np.concatenate(
                (
                    np.ones((self.ksp, 1)) * self.yinit,
                    np.ones((del_sp, 1)) * 1.0,
                )
            )
        else:
            self.r = np.concatenate(
                (
                    np.ones((self.ksp, 1)) * self.yinit,
                    np.ones((del_sp, 1)) * np.random.uniform(sp_min, sp_max),
                    np.ones((del_sp, 1)) * np.random.uniform(sp_min, sp_max),
                    np.ones((del_sp, 1)) * np.random.uniform(sp_min, sp_max),
                )
            )

        sim_time = len(self.r) * self.delt
        self.ttfinal = (
            self.ttfinal
            if self.ttfinal is not None and self.ttfinal < sim_time
            else sim_time
        )
        self.tt = np.arange(0, self.ttfinal, self.delt)  # time vector
        self.kfinal = len(self.tt)  # number of time intervals
        return self.reset_input()

    def reset_input(self):
        # Input vector
        self.input = np.zeros((self.kfinal + 1, self.n_actions))
        self.input[: self.ksp] = np.ones((self.ksp, 1)) * self.uinit
        return self.reset_env()

    def reset_env(self):
        # Reset sim time
        self.k = self.ksp - 1

        #  Control vector
        self.u = np.zeros((self.kfinal + 1, 1))
        self.u[: self.ksp] = np.ones((self.ksp, 1)) * self.uinit
        self.du = np.zeros((self.kfinal, 1))

        # Environment State vector
        self.x = np.zeros((self.kfinal + 1, self.sysd_plant.A.shape[0], 1))
        self.x[: self.ksp] = np.repeat(np.array(self.xinit), self.ksp).reshape(
            self.ksp, self.sysd_plant.A.shape[0], 1
        )

        # Output vector
        self.y = np.zeros((self.kfinal + 1, 1))
        self.y[: self.ksp] = np.ones((self.ksp, 1)) * self.yinit

        # Error vector
        self.E = np.zeros((self.kfinal, 1))
        return self.r[self.k][0], self.y[self.k][0]

    def step_env(self, u):
        self.E[self.k] = self.r[self.k][0] - self.y[self.k][0]
        # Slew rate
        if self.slew_rate:
            u = np.clip(
                u,
                self.u[self.k - 1] - self.slew_rate,
                self.u[self.k - 1] + self.slew_rate,
            )
        # Disturbance
        d = self.disturbance_value * np.random.randn() if self.disturbance else 0.0
        self.u[self.k] = u + d
        # Control Constraints
        self.u[self.k] = np.clip(self.u[self.k], self.umin, self.umax)
        #  plant equations
        self.x[self.k + 1] = (
            np.matmul(self.phi, self.x[self.k])
            + np.matmul(self.gamma, self.u[self.k]).T
        )
        self.y[self.k + 1] = np.matmul(self.cd, self.x[self.k + 1])

        self.k = self.k + 1
        return self.r[self.k][0], self.y[self.k][0]

    def step(self, u):
        self.input[self.k] = np.array(u)
        return self.step_env(u)

    def get_state(self):
        return self.r[self.k][0], self.y[self.k][0], self.y[self.k - 1][0]

    def get_axis(self, use_sample_instant=True):
        axis = self.tt[: self.k].copy()
        axis_name = "Time (min)"
        if use_sample_instant:
            axis = np.arange(self.k)
            axis_name = "Sampling Instants"
        return axis, axis_name

    def plot(self, save=False, use_sample_instant=True):
        axis, axis_name = self.get_axis(use_sample_instant)
        plt.figure(figsize=(16, 20))
        plt.subplot(3, 1, 1)
        plt.step(
            axis, self.r[: self.k], linestyle="dashed", label="Setpoint", where="post"
        )
        plt.plot(axis, self.y[: self.k], label="Plant Output")
        plt.ylabel("")
        plt.xlabel(axis_name)
        ise = f"{self.ise():.3e}"
        title = f"ISE: {ise}"
        plt.title(title)
        plt.xlim(axis[0], axis[-1])
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.step(axis, self.u[: self.k], label="Control Input", where="post")
        plt.ylabel("")
        plt.xlabel(axis_name)
        plt.title("Control Action")
        plt.xlim(axis[0], axis[-1])
        plt.grid()
        plt.legend()

        plt.subplot(3, 1, 3)
        for i in range(self.n_actions):
            plt.plot(
                axis[self.ksp :],
                self.input[self.ksp : self.k, i],
                label=self.input_names[i],
            )
        plt.ylabel("Value")
        plt.xlabel(axis_name)
        plt.title("Inputs")
        plt.xlim(axis[0], axis[-1])
        plt.grid()
        plt.legend()
        if save:
            plt.tight_layout()
            img = fig2data(plt.gcf())
            plt.close()
            return img

    def ise(self):
        return ((self.r[: self.k] - self.y[: self.k]) ** 2).sum()

    def iae(self):
        return abs(self.r[: self.k] - self.y[: self.k]).sum()


min_gains = [0.0, 0.0, 0.0]
max_gains = [20.0, 2.0, 2.0]


class SystemSimplePID(System):
    @property
    def input_names(self):
        return ["Kp(k)", "Ki(k)", "Kd(k)"]

    def reset_input(self, *args, **kwargs):
        auto = False
        self.Gc = PID(
            5,
            1,
            0,
            setpoint=self.yinit,
            sample_time=self.delt,
            output_limits=(self.umin, self.umax),
            auto_mode=auto,
        )
        self.Gc.set_auto_mode(not auto, last_output=self.uinit)
        self.slew_rate = None
        self.input_low = np.array(min_gains)
        self.input_high = np.array(max_gains)
        # Input vector
        self.input = np.zeros((self.kfinal + 1, self.n_actions))
        self.input[: self.ksp] = np.ones((self.ksp, self.n_actions)) * np.array(
            [5, 1, 0]
        )
        self.gains = []
        self.gain_components = []
        return self.reset_env(*args, **kwargs)

    def step(self, Kp, taui, taud):
        Ki = Kp / (taui + 0.01)
        Kd = Kp * taud
        self.Gc.setpoint = self.r[self.k][0]
        self.Gc.tunings = (Kp, Ki, Kd)
        u = self.Gc(self.y[self.k][0], self.delt)

        self.input[self.k] = np.array([Kp, Ki, Kd])
        self.gains.append([Kp, taui, taud])
        self.gain_components.append(self.Gc.components)
        return self.step_env(u)

    def plot_gains(self, save=False, use_sample_instant=True):
        axis, axis_name = self.get_axis(use_sample_instant)
        plt.figure(figsize=(16, 12))
        labels = ["$K_p$", "tau_I", "tau_D"]
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(
                axis[self.ksp : self.ksp + len(self.gains)],
                np.array(self.gains)[:-1, i],
                label=labels[i],
            )
            plt.ylabel("Value")
            plt.xlabel(axis_name)
            plt.xlim(axis[0], axis[-1])
            plt.grid()
            plt.legend()
        if save:
            plt.tight_layout()
            img = fig2data(plt.gcf())
            plt.close()
            return img

    def plot_actual_gains(self, save=False, use_sample_instant=True):
        axis, axis_name = self.get_axis(use_sample_instant)
        plt.figure(figsize=(16, 12))
        labels = ["$K_p$", "$K_I$", "$K_D$"]
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(
                axis[self.ksp :],
                np.array(self.input)[self.ksp : self.k, i],
                label=labels[i],
            )
            plt.ylabel("Value")
            plt.xlabel(axis_name)
            plt.xlim(axis[0], axis[-1])
            plt.grid()
            plt.legend()
        if save:
            plt.tight_layout()
            img = fig2data(plt.gcf())
            plt.close()
            return img

    def plot_gain_components(self, use_sample_instant=True):
        axis, axis_name = self.get_axis(use_sample_instant)
        plt.figure(figsize=(16, 9))
        labels = ["Proportional", "Integral", "Derivative"]
        for i in range(3):
            plt.subplot(3, 1, i + 1)
            plt.plot(
                axis[self.ksp : self.ksp + len(self.gain_components)],
                np.array(self.gain_components)[:-1, i],
                label=labels[i],
            )
            plt.ylabel("Value")
            plt.xlabel(axis_name)
            plt.xlim(axis[0], axis[-1])
            plt.grid()
            plt.legend()

class GymSystem(gym.Env):
    def __init__(
        self,
        uinit=uinit,
        yinit=yinit,
        num=num,
        dem=dem,
        system=SystemSimplePID,
        disturbance=disturbance,
        deterministic=deterministic,
        disturbance_value=disturbance_value,
        reward_mode="m-PPO"
    ):
        super().__init__()
        self.reward_mode = reward_mode

        self.uinit = uinit
        self.yinit = yinit
        self.num = num
        self.dem = dem
        self.disturbance = disturbance
        self.deterministic = deterministic
        self.disturbance_value = disturbance_value
        self.system = system(
            uinit=self.uinit,
            yinit=self.yinit,
            num=self.num,
            dem=self.dem,
            disturbance=self.disturbance,
            deterministic=self.deterministic,
            disturbance_value=self.disturbance_value,
        )

        self.n_actions = self.system.n_actions
        self.action_space = spaces.Box(-1.0, 1.0, (self.n_actions,))
        self.n_states = self.system.n_states
        self.observation_space = spaces.Box(
            low=-100.0, high=100.0, shape=(self.n_states,), dtype=np.float32
        )

    def step(self, action, debug=False):
        action = self.convert_action(action)
        obs = self.system.step(*action)
        
        # --- UPDATE THIS LINE TO PASS THE MODE ---
        reward = self.get_reward(obs, mode=self.reward_mode) 
        # -----------------------------------------
        
        done = bool(self.system.k == self.system.kfinal - 1)
        return self.convert_state(), reward, done, {}

    def convert_state(self):
        obs = self.system.get_state()
        obs = np.array(obs).astype(np.float32)
        return obs

    def convert_action(self, action):
        actions = (action + 1) * (
            self.system.input_high - self.system.input_low
        ) * 0.5 + self.system.input_low
        actions = np.clip(actions, self.system.input_low, self.system.input_high)
        return actions

    def unconvert_action(self, action):
        actions = (2.0 * action - (self.system.input_high + self.system.input_low)) / (
            self.system.input_high - self.system.input_low
        )
        actions = np.clip(actions, -1.0, 1.0)
        return actions

    def reset(self):
        _ = self.system.reset()
        obs = self.convert_state()
        return obs

    def get_reward(self, obs, mode="m-PPO"):
        e = obs[0] - obs[1] # Setpoint - Output
        
        if mode == "m-PPO":
            # The Paper's Strategy: Clipped Squared Error + Bonus 
            eta = 0.01  # Scaling factor 
            xi = 5.0    # Clipping factor 
            r_ise = -np.minimum(np.maximum(eta * (e**2), 0), xi)
            
            # Binary bonus (r_add) for high precision 
            r_add = 2.0 if np.abs(e) < 0.01 else 0.0
            reward = r_ise + r_add
            return reward

        elif mode == "Standard_MSE":
            # Failure Case 1: Raw Squared Error (Common in DDPG) 
            # Predicted: Gradients will explode as 'e' grows 
            reward = -(e**2)
            return reward

        elif mode == "Standard_MAE":
            # Failure Case 2: Linear Absolute Error (Common in A2C) 
            # Predicted: High oscillations in the unstable region 
            reward = -np.abs(e)
            return reward

        elif mode == "ITAE":
            # Failure Case 3: Time-weighted Error
            # Predicted: Extreme penalties late in the episode cause divergence.
            reward = -(self.system.k * self.system.delt) * np.abs(e)
            return reward

        elif mode == "Sparse":
            # Failure Case 4: Binary success/failure
            # Predicted: Agent will never find the setpoint.
            reward =  1.0 if np.abs(e) < 0.01 else -0.1
            return reward

    def step(self, action, debug=False):
        # sat_act = np.sum(action[action > 0.96]) + np.sum(action[action < -0.96])
        if debug:
            print("Original: ", action)
        action = self.convert_action(action)
        if debug:
            print("Converted: ", action)
        obs = self.system.step(*action)
        reward = self.get_reward(obs, mode=self.reward_mode)
        done = bool(self.system.k == self.system.kfinal - 1)
        info = {}
        obs = self.convert_state()
        return obs, reward, done, info

    def render(self, mode="human"):
        if mode == "human":
            print("ISE: ", self.system.ise())
            self.system.plot()
        elif mode == "rgb_array":
            return self.system.plot(save=True)

    def close(self):
        pass


import io
import os

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
from control.matlab import *
from gym import spaces
from scipy.integrate import solve_ivp


class ActionRepeat(gym.Wrapper):
    def __init__(self, env, amount=1):
        super().__init__(env)
        self.amount = amount

    def step(self, action):
        done = False
        total_reward = 0
        current_step = 0
        while current_step < self.amount and not done:
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            current_step += 1
        return obs, total_reward, done, info


class EarlyStopping(gym.Wrapper):
    def __init__(self, env, y_lim=[-100, 100]):
        super().__init__(env)
        self.y_lim = y_lim

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if (
            self.env.system.y[self.env.system.k] > self.y_lim[1]
            or self.env.system.y[self.env.system.k] < self.y_lim[0]
        ):
            done = True
            reward += -20.0
        return obs, reward, done, info


import argparse
import os

import stable_baselines3
import torch
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecCheckNan, VecNormalize
from stable_baselines3.common.noise import NormalActionNoise

from callbacks import EvalCallback, SaveBestModelCallback

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="SystemSimplePID")
parser.add_argument("--algo", default="PPO")
parser.add_argument("--logdir", default="logs")
parser.add_argument("--action_repeat", type=int, default=2)
parser.add_argument("--vec_normalize", default="True")
parser.add_argument("--early_stopping", default="True")
parser.add_argument("--mode", default="train")
parser.add_argument("--reward_type", default="m-PPO", 
                    choices=["m-PPO", "Standard_MSE", "Standard_MAE", "ITAE", "Sparse"])
args = parser.parse_args()

env_model = args.model
env_class = {
    "SystemSimplePID": SystemSimplePID,
}[env_model]
print(env_class)

torch.autograd.set_detect_anomaly(True)
print("CUDA Available: ", torch.cuda.is_available())

early_stopping = args.early_stopping == "True"
print("Using Early Stopping: ", early_stopping)

action_repeat = True
action_repeat_value = args.action_repeat
print("Using Action Repeat: ", action_repeat, action_repeat_value)

use_sde = False
print("Using gSDE: ", use_sde)

vec_normalize = args.vec_normalize == "True"
print("Using VecNormalize: ", vec_normalize)

algo = args.algo
print("Algorithm: ", algo)

extra = "BetterES_SystemFix"

tag_name = f"CS1_{env_model}_{algo}_AR_{action_repeat}_{action_repeat_value}_use_sde_{use_sde}_ES_{early_stopping}_extra_{extra}"
print("Run Name: ", tag_name)

base_log = args.logdir
log_dir = os.path.join(base_log, "CS1", tag_name)

save_callback = SaveBestModelCallback(
    check_freq=20000, log_dir=log_dir, verbose=1
)

eval_env = GymSystem(system=env_class, reward_mode=args.reward_type)
if early_stopping:
    eval_env = EarlyStopping(eval_env)
if action_repeat:
    eval_env = ActionRepeat(eval_env, action_repeat_value)
save_image_callback = EvalCallback(
    eval_env=eval_env, eval_freq=50000, log_dir=None, name="Random"
)

eval_env2 = GymSystem(system=env_class, deterministic=True, reward_mode=args.reward_type)
if early_stopping:
    eval_env2 = EarlyStopping(eval_env2)
if action_repeat:
    eval_env2 = ActionRepeat(eval_env2, action_repeat_value)
save_image_callback2 = EvalCallback(
    eval_env=eval_env2, eval_freq=50000, log_dir=log_dir, name="Deterministic"
)

callback = CallbackList([save_callback, save_image_callback, save_image_callback2])
print(callback.callbacks)


env = GymSystem(system=env_class, reward_mode=args.reward_type)
if early_stopping:
    env = EarlyStopping(env)
if action_repeat:
    env = ActionRepeat(env, action_repeat_value)
'env = make_vec_env(lambda: env, n_envs=1, monitor_dir=log_dir)'
# Protect our training logs! Only write a monitor file if we are training.
if args.mode == "train":
    env = make_vec_env(lambda: env, n_envs=1, monitor_dir=log_dir)
else:
    env = make_vec_env(lambda: env, n_envs=1, monitor_dir=None)
if vec_normalize:
    if os.path.exists(os.path.join(log_dir, "vec_normalize.pkl")):
        print("Found VecNormalize Stats. Using stats")
        env = VecNormalize.load(os.path.join(log_dir, "vec_normalize.pkl"), env)
    else:
        print("No previous stats found. Using new VecNormalize instance.")
        env = VecNormalize(env)
else:
    env.normalize_obs = lambda x: x
env = VecCheckNan(env, raise_exception=True)

'''algo_class = getattr(stable_baselines3, algo)
model = algo_class("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)'''

algo_class = getattr(stable_baselines3, algo)

if algo == "DDPG":
    # DDPG requires manual action noise for exploration
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    model = algo_class("MlpPolicy", env, action_noise=action_noise, verbose=1, tensorboard_log=log_dir, batch_size=256)

elif algo == "SAC":
    # SAC works best with specific batch sizes for continuous control
    model = algo_class("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, batch_size=256)

else:
    # PPO and A2C run perfectly with standard defaults
    model = algo_class("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

best_model_path = os.path.join(log_dir, "best_model.zip")
if os.path.exists(best_model_path) or args.mode == "test":
    assert os.path.exists(best_model_path), f"Path doesnt exist: {best_model_path}"
    print(f"Found previous checkpoint. Loading from checkpoint. {best_model_path}")
    model = algo_class.load(best_model_path, env)
print(model)

if args.mode == "train":
    tsteps = 500_000
    model.learn(tsteps, reset_num_timesteps=False, callback=callback)


save_path = Path(log_dir).parts[-2:]
save_path = os.path.join(*save_path)
test_log_dir = os.path.join("..", "results", save_path, "test_files", "servo")
os.makedirs(test_log_dir, exist_ok=True)

test_env = GymSystem(system=env_class, disturbance=False, deterministic=True, reward_mode=args.reward_type)
if action_repeat:
    test_env = ActionRepeat(test_env, action_repeat_value)
evaluate(model, test_env, action_repeat_value, test_log_dir)

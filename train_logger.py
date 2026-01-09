import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor  # <--- The key ingredient
from bio_env import BioProstheticWrapper
import os

# SETTINGS
TIMESTEPS = 80000 
SEED = 42

# Setup Environment with Monitoring
def make_env(algo_name):
    env = gym.make("Pendulum-v1")
    env = BioProstheticWrapper(env, jerk_penalty_weight=0.1)
    # This saves the data for the learning curve
    env = Monitor(env, filename=f"{algo_name}_log") 
    return env

# 1. Train DDPG
print("Generating DDPG Learning Curve Data...")
env_ddpg = DummyVecEnv([lambda: make_env("ddpg")])
n_actions = env_ddpg.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model_ddpg = DDPG("MlpPolicy", env_ddpg, action_noise=action_noise, seed=SEED, verbose=0)
model_ddpg.learn(total_timesteps=TIMESTEPS)
print("DDPG Done.")

# 2. Train TD3
print("Generating TD3 Learning Curve Data...")
env_td3 = DummyVecEnv([lambda: make_env("td3")])
model_td3 = TD3("MlpPolicy", env_td3, action_noise=action_noise, seed=SEED, verbose=0)
model_td3.learn(total_timesteps=TIMESTEPS)
print("TD3 Done. Ready to plot.")

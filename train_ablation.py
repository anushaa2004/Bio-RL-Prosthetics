import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from bio_env import BioProstheticWrapper

# SETTINGS
TIMESTEPS = 80000 
SEED = 42

# 1. Setup Environment WITH ZERO PENALTY (Standard control)
def make_env():
    env = gym.make("Pendulum-v1")
    # We set penalty to 0.0 to see how the "raw" AI behaves
    env = BioProstheticWrapper(env, jerk_penalty_weight=0.0) 
    return env

env_raw = DummyVecEnv([make_env])

# 2. Setup Noise
n_actions = env_raw.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# 3. Train
print("Training 'Raw' TD3 (No Penalty) for comparison...")
model_raw = TD3("MlpPolicy", env_raw, action_noise=action_noise, seed=SEED, verbose=1)
model_raw.learn(total_timesteps=TIMESTEPS)
model_raw.save("td3_raw_model")
print("Done! Now you have a baseline.")

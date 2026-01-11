import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from bio_env import BioProstheticWrapper

# SETTINGS
TIMESTEPS = 80000 
N_SEEDS = 5  # <--- MATCHING THE MAIN EXPERIMENT

# Ensure logs folder exists
os.makedirs("logs", exist_ok=True)

def make_env(seed):
    env = gym.make("Pendulum-v1")
    # Zero penalty = "Raw" / "Unconstrained"
    env = BioProstheticWrapper(env, jerk_penalty_weight=0.0) 
    
    # Save as td3_raw_seed_0, td3_raw_seed_1, etc.
    log_name = f"logs/td3_raw_seed_{seed}"
    env = Monitor(env, filename=log_name) 
    return env

for seed in range(N_SEEDS):
    print(f"\n--- Training Raw TD3 (Seed {seed+1}/{N_SEEDS}) ---")
    
    # 1. Setup Env
    env_raw = DummyVecEnv([lambda: make_env(seed)])
    
    # 2. Setup Noise
    n_actions = env_raw.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # 3. Train
    model_raw = TD3("MlpPolicy", env_raw, action_noise=action_noise, seed=seed, verbose=1)
    model_raw.learn(total_timesteps=TIMESTEPS)
    
    # Save the first one just in case you need a model file later
    if seed == 0:
        model_raw.save("td3_raw_model")

print("Ablation Training Complete.")

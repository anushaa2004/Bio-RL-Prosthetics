import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from bio_env import BioProstheticWrapper

# SETTINGS
TIMESTEPS = 80000 
N_SEEDS = 5  # Run 5 times for error bands (Goal A)

def make_env(algo_name, seed):
    env = gym.make("Pendulum-v1")
    env = BioProstheticWrapper(env, jerk_penalty_weight=0.1)
    # Save separate logs for each seed: td3_log_0, td3_log_1...
    log_name = f"logs/{algo_name}_seed_{seed}"
    os.makedirs("logs", exist_ok=True)
    return Monitor(env, filename=log_name)

# We only need checkpoints for ONE run to do the histograms (Goal B)
checkpoint_callback = CheckpointCallback(
    save_freq=20000, 
    save_path='./checkpoints/',
    name_prefix='td3_bio'
)

for seed in range(N_SEEDS):
    print(f"--- Training Run {seed+1}/{N_SEEDS} ---")
    
    # 1. DDPG
    env_ddpg = DummyVecEnv([lambda: make_env("ddpg", seed)])
    n_actions = env_ddpg.action_space.shape[-1]
    noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    model_ddpg = DDPG("MlpPolicy", env_ddpg, action_noise=noise, seed=seed, verbose=0)
    model_ddpg.learn(total_timesteps=TIMESTEPS)
    
    # 2. TD3
    env_td3 = DummyVecEnv([lambda: make_env("td3", seed)])
    model_td3 = TD3("MlpPolicy", env_td3, action_noise=noise, seed=seed, verbose=0)
    
    # Only save checkpoints for the first seed to save disk space
    if seed == 0:
        model_td3.learn(total_timesteps=TIMESTEPS, callback=checkpoint_callback)
        model_td3.save("td3_bio_model_final") # Save final for later use
    else:
        model_td3.learn(total_timesteps=TIMESTEPS)

print("Training Complete. Logs in /logs folder. Checkpoints in /checkpoints.")

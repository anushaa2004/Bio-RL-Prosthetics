import gymnasium as gym
import numpy as np
from stable_baselines3 import TD3, DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from bio_env import BioProstheticWrapper
import os

# SETUP
TIMESTEPS = 80000 
SEED = 42

# HELPER FUNCTION
# This allows us to create multiple environments cleanly if needed
def make_env(jerk_weight=0.1):
    env = gym.make("Pendulum-v1")
    env = BioProstheticWrapper(env, jerk_penalty_weight=jerk_weight)
    return env

# 1. Create Vectorized Environments (Standard SB3 practice)
# We wrap them in DummyVecEnv which is required for some SB3 features
env_ddpg = DummyVecEnv([lambda: make_env(jerk_weight=0.1)])
env_td3 = DummyVecEnv([lambda: make_env(jerk_weight=0.1)])

# 2. Create Action Noise
# DDPG and TD3 need noise to explore! 
# We use Gaussian noise (mean=0, sigma=0.1)
n_actions = env_ddpg.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# 3. Define Models with Noise
print("Initializing DDPG...")
model_ddpg = DDPG("MlpPolicy", env_ddpg, action_noise=action_noise, seed=SEED, verbose=1)

print("Initializing TD3...")
model_td3 = TD3("MlpPolicy", env_td3, action_noise=action_noise, seed=SEED, verbose=1)

# 4. Train DDPG
print("Starting DDPG Training...")
model_ddpg.learn(total_timesteps=TIMESTEPS)
model_ddpg.save("ddpg_bio_model")
print("DDPG Training Finished.")

# 5. Train TD3
print("Starting TD3 Training...")
model_td3.learn(total_timesteps=TIMESTEPS)
model_td3.save("td3_bio_model")
print("TD3 Training Finished.")

print("ALL DONE! Models saved.")

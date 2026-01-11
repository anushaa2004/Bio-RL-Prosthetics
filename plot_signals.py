import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3, DDPG
from bio_env import BioProstheticWrapper

# LOAD MODELS
print("Loading models...")
model_ddpg = DDPG.load("ddpg_bio_model")
model_td3_bio = TD3.load("td3_bio_model")   # Your good model (with penalty)
try:
    model_td3_raw = TD3.load("td3_raw_model") # The new one (no penalty)
except:
    model_td3_raw = None # Handle case if you haven't run ablation yet

# SETUP TEST
env = BioProstheticWrapper(gym.make("Pendulum-v1"), jerk_penalty_weight=0.0)
obs, _ = env.reset(seed=42)

# RECORD ONE EPISODE
actions_ddpg = []
actions_td3 = []
actions_raw = []

# We run the simulation identically for all agents
print("Simulating episodes...")
for step in range(100): # Just 100 steps (half a second) to zoom in
    # DDPG
    act_d, _ = model_ddpg.predict(obs, deterministic=True)
    actions_ddpg.append(act_d[0])
    
    # TD3 (Bio)
    act_t, _ = model_td3_bio.predict(obs, deterministic=True)
    actions_td3.append(act_t[0])
    
    # TD3 (Raw) - Optional
    if model_td3_raw:
        act_r, _ = model_td3_raw.predict(obs, deterministic=True)
        actions_raw.append(act_r[0])
        
    # Just step randomly to keep the physics moving same for all (approx)
    obs, _, _, _, _ = env.step(act_t)

# PLOT
plt.figure(figsize=(10, 5))
plt.plot(actions_ddpg, label="DDPG (Standard)", color="green", linestyle="--", alpha=0.7)
if model_td3_raw:
    plt.plot(actions_raw, label="TD3 (No Penalty)", color="blue", alpha=0.5)
plt.plot(actions_td3, label="TD3 (Bio-Constrained)", color="orange", linewidth=2.5)

plt.title("Actuator Control Signal Analysis (First 100 Steps)")
plt.xlabel("Time Step (ms)")
plt.ylabel("Motor Voltage / Torque")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

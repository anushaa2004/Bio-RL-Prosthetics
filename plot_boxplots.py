import gymnasium as gym
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from stable_baselines3 import TD3, DDPG
from bio_env import BioProstheticWrapper

# SETTINGS
N_EPISODES = 30  # Test on 30 different "patients"
MAX_STEPS = 200

def run_test(model, env_name="Pendulum-v1", algo_name="Unknown"):
    """
    Runs the model on the environment and tracks Reward & Jerk.
    """
    # Create a fresh bio-environment
    env = BioProstheticWrapper(gym.make(env_name), jerk_penalty_weight=0.1, randomize_patient=True)
    
    results = []
    
    print(f"Testing {algo_name} on {N_EPISODES} different patients...")
    
    for i in range(N_EPISODES):
        obs, _ = env.reset()
        total_reward = 0
        total_jerk = 0
        
        for _ in range(MAX_STEPS):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            total_reward += reward
            # We recover the raw jerk from the info dict we made
            total_jerk += info.get('jerk', 0)
            
            if terminated or truncated:
                break
        
        results.append({
            "Algorithm": algo_name,
            "Total Reward": total_reward,
            "Total Jerk (Smoothness Cost)": total_jerk
        })
        
    env.close()
    return pd.DataFrame(results)

# 1. Load the Models
# We assume the files are in the same folder
print("Loading models...")
try:
    model_ddpg = DDPG.load("ddpg_bio_model")
    model_td3 = TD3.load("td3_bio_model")
except FileNotFoundError:
    print("ERROR: Could not find model files. Did you save them in this folder?")
    exit()

# 2. Run the Tests
df_ddpg = run_test(model_ddpg, algo_name="DDPG (Standard)")
df_td3 = run_test(model_td3, algo_name="TD3 (Conservative)")

# Combine data
full_data = pd.concat([df_ddpg, df_td3])

# 3. Plotting
# We create two side-by-side boxplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Reward (Performance)
sns.boxplot(data=full_data, x="Algorithm", y="Total Reward", ax=axes[0], palette="Set2")
axes[0].set_title("Clinical Performance (Higher is Better)")
axes[0].set_ylabel("Return (Reward)")

# Plot 2: Jerk (Safety)
sns.boxplot(data=full_data, x="Algorithm", y="Total Jerk (Smoothness Cost)", ax=axes[1], palette="Set2")
axes[1].set_title("Actuator Stress/Jerk (Lower is Safer)")
axes[1].set_ylabel("Cumulative Jerk")

print("Displaying plots...")
plt.tight_layout()
plt.show()


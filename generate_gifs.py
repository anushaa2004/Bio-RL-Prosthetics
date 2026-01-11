import imageio
import numpy as np
import gymnasium as gym
from stable_baselines3 import DDPG, TD3
from stable_baselines3.common.noise import NormalActionNoise
from bio_env import BioProstheticWrapper 
import os

# --- HARDCODED PATH TO YOUR Z: DRIVE ---
BASE_PATH = r"Z:\Anusha\Imperial\Year 4\Reinforcement Learning"

def get_model_or_train(algo_name, model_filename):
    model_path = os.path.join(BASE_PATH, model_filename)
    
    # 1. Setup Environment
    env = gym.make("Pendulum-v1", render_mode="rgb_array")
    env = BioProstheticWrapper(env, jerk_penalty_weight=0.1)

    # 2. Try to Load Existing Model
    try:
        print(f"🔍 Attempting to load {algo_name} from {model_path}...")
        if algo_name == "DDPG":
            model = DDPG.load(model_path, env=env)
        else:
            model = TD3.load(model_path, env=env)
        print(f"✅ Loaded {algo_name} successfully!")
        return model, env
    except Exception as e:
        print(f"⚠️ Could not load model (Version mismatch or missing). Re-training {algo_name}...")
    
    # 3. If Load Fails, Re-Train Automatically
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
    
    if algo_name == "DDPG":
        model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)
    else:
        model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)
    
    # Fast training just for the GIF
    model.learn(total_timesteps=10000) 
    model.save(model_path) # Save it so we have it for next time
    print(f"💾 Re-trained and saved {algo_name}!")
    return model, env

def record_gif(algo_name, model_filename, gif_filename):
    # Get the model (load or train)
    model, env = get_model_or_train(algo_name, model_filename)
    
    output_path = os.path.join(BASE_PATH, gif_filename)
    print(f"🎥 Recording GIF for {algo_name}...")
    
    obs, _ = env.reset(seed=42)
    images = []
    
    # Record 150 frames
    for _ in range(150):
        images.append(env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    imageio.mimsave(output_path, images, fps=30)
    print(f"🎉 Saved GIF to: {output_path}")
    env.close()

# --- RUN EVERYTHING ---
if __name__ == "__main__":
    record_gif("DDPG", "ddpg_bio_model", "ddpg_jitter.gif")
    record_gif("TD3",  "td3_bio_model",  "td3_smooth.gif")

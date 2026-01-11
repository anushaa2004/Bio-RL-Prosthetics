import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import TD3
from bio_env import BioProstheticWrapper

# 1. Load the Best Model (80k steps)
# Make sure this file exists in your folder!
model_path = "td3_bio_model_final" 
try:
    model = TD3.load(model_path)
    print("✅ Model loaded successfully.")
except:
    print("❌ Model not found. Make sure 'td3_bio_model_final.zip' is in this folder.")
    exit()

# 2. Setup Grid
print("Generating Heatmap...")
# Angle: -180 to +180 degrees (in radians)
th_vals = np.linspace(-np.pi, np.pi, 200)
# Velocity: -8 to +8
thdot_vals = np.linspace(-8, 8, 200)

grid_actions = np.zeros((len(thdot_vals), len(th_vals)))

# 3. Predict across the grid
for i, th in enumerate(th_vals):
    for j, thdot in enumerate(thdot_vals):
        # Pendulum Observation: [cos(theta), sin(theta), theta_dot]
        obs = np.array([np.cos(th), np.sin(th), thdot])
        
        # Get action from the agent
        action, _ = model.predict(obs, deterministic=True)
        grid_actions[j, i] = action[0]

# 4. Plot
plt.figure(figsize=(9, 7))

# Create the heatmap
plt.imshow(
    grid_actions, 
    extent=[-180, 180, -8, 8], # Axis labels in Degrees/Velocity
    origin='lower', 
    cmap='coolwarm', # Blue = Negative Torque, Red = Positive Torque
    aspect='auto',
    vmin=-2.0, vmax=2.0 # Cap colors at max torque
)

plt.colorbar(label="Motor Torque (Nm)")
plt.xlabel("Joint Angle (Degrees)")
plt.ylabel("Angular Velocity (rad/s)")
plt.title("Learned Control Policy (Bio-Constrained)")

# Draw crosshairs at 0,0
plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')

plt.tight_layout()
plt.savefig("policy_heatmap.png", dpi=300)
print("✅ Heatmap saved as 'policy_heatmap.png'")
plt.show()

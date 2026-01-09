import pandas as pd
import matplotlib.pyplot as plt

def plot_log(filename, label, color):
    try:
        # Skip the first 2 rows (header junk from SB3)
        df = pd.read_csv(filename + ".monitor.csv", skiprows=1)
        # Smooth the curve so it looks nice (Moving Average)
        return df['r'].rolling(window=50).mean()
    except:
        return None

plt.figure(figsize=(10, 5))

# Plot DDPG
ddpg_data = plot_log("ddpg_log", "DDPG (Standard)", "green")
if ddpg_data is not None:
    plt.plot(ddpg_data, label="DDPG (Standard)", color="#66c2a5", alpha=0.8)

# Plot TD3
td3_data = plot_log("td3_log", "TD3 (Conservative)", "orange")
if td3_data is not None:
    plt.plot(td3_data, label="TD3 (Conservative)", color="#fc8d62", linewidth=2)

plt.title("Learning Efficiency: TD3 vs DDPG")
plt.xlabel("Episode")
plt.ylabel("Reward (Moving Average)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os

def get_data(algo_prefix, label):
    if not os.path.exists("logs"):
        print("Folder 'logs' not found.")
        return None
        
    # Matches logs/td3_seed_0, logs/td3_raw_seed_0, etc.
    files = glob.glob(f"logs/{algo_prefix}_seed_*.monitor.csv")
    
    if not files:
        print(f"No files found for {algo_prefix}")
        return None

    data_frames = []
    for f in files:
        try:
            df = pd.read_csv(f, skiprows=1)
            df['time'] = df.index 
            data_frames.append(df)
        except Exception as e:
            print(f"Skipping bad file {f}: {e}")
        
    if not data_frames: return None
    
    # Combine and ignore index to fix the plotting error
    combined = pd.concat(data_frames, ignore_index=True)
    combined['Algorithm'] = label
    return combined

plt.figure(figsize=(10, 6))

# 1. Load Data (Now all 3 use the same robust function)
df_ddpg = get_data("ddpg", "DDPG (Standard)")
df_td3  = get_data("td3", "TD3 (Conservative)")
df_raw  = get_data("td3_raw", "TD3 (No Penalty)") # <--- NOW ROBUST

# 2. Combine
data_list = []
if df_ddpg is not None: data_list.append(df_ddpg)
if df_td3 is not None:  data_list.append(df_td3)
if df_raw is not None:  data_list.append(df_raw)

if data_list:
    combined_all = pd.concat(data_list, ignore_index=True)

    # Smoothing
    combined_all['Reward'] = combined_all.groupby('Algorithm')['r'].transform(lambda x: x.rolling(50).mean())

    # Plot
    # This will now show SHADING for all 3 lines
    sns.lineplot(data=combined_all, x="time", y="Reward", hue="Algorithm", errorbar='sd')

    plt.title("Learning Efficiency (Averaged over 5 Seeds)")
    plt.xlabel("Episode")
    plt.ylabel("Reward (Smoothed)")
    plt.grid(True, alpha=0.3)
    plt.show()
else:
    print("No data found! Run train_logger.py and train_ablation.py first.")

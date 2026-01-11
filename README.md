# Bio-RL-Prosthetics

This repository contains the code for the tutorial **"From Jittery Joints to Smooth Control: TD3 for Robotic Prosthetics."**

It demonstrates how to use **Twin Delayed DDPG (TD3)** with a custom **Bio-Prosthetic Wrapper** to solve the "Safety Gap" in reinforcement learning for biomechanics.

#### Key Features
* **BioProstheticWrapper:** A custom Gym wrapper that penalises high-frequency torque changes (jerk).
* **Comparison:** Benchmarks TD3 (Conservative) against DDPG (Optimistic).
* **Analysis:** Includes scripts to visualize actuator voltage, variance, and clinical safety metrics.

## Installation
Install the dependencies:
   ```bash
pip install -r requirements.txt
```
## Usage
1) To train the agents (Single Run):

```bash
python train.py
```
This trains both DDPG and TD3 agents for 80,000 steps. Saves ddpg_bio_model.zip and td3_bio_model.zip.

To generate the "Raw" (No Penalty) baseline for signal comparison:

```bash
python train_ablation.py
```

2) To generate the statistical data (mean ± std dev) for the learning curves:

```bash
python train_experiment.py
python train_experiment_ablation.py
```
This runs the training across 5 random seeds to verify stability.

## Reproducing Figures

Once training is complete, you can reproduce every figure from the tutorial:

Figure 3:
```bash
python plot_learning_curve.py
```
Plots Learning Efficiency (Reward over time) with error bands.

Figure 4:
```bash
python plot_boxplots.py
```
Compares Clinical Performance vs. Actuator Stress (Jerk).

Figure 5:
```bash
python generate_gifs.py
```
Saves animations of the "Jittery" DDPG vs "Smooth" TD3 agents.

Figure 6:
```bash
python plot_signals.py
```	
Visualises the raw Actuator Control Signals (Voltage).

Figure 7:
```bash
python plot_heatmap.py	
```
Generates the Policy Heatmap showing the energy-saving dead zone.


## File Structure

### Core Logic
* `bio_env.py`: Custom gym wrapper that implements the biomechanical constraints (jerk penalty).

### Training Scripts
* `train_models.py`: Main script to train single DDPG and TD3 agents (creates `.zip` files).
* `train_experiment.py`: Runs the 5-seed experiment for DDPG and TD3 (creates logs for Fig 3).
* `train_ablation.py`: Trains the "Raw" TD3 baseline (no bio-constraints).

### Visualisation Scripts
* `plot_learning_curve.py`: Generates the learning efficiency graph.
* `plot_boxplots.py`: Generates the safety boxplots.
* `plot_signals.py`: Plots voltage control signals.
* `plot_heatmap.py`: Plots the policy action heatmap.
* `generate_gifs.py`: Records .gif animations of the agents.

# Bio-RL-Prosthetics

This repository contains the code for the tutorial **"From Jittery Joints to Smooth Control: TD3 for Robotic Prosthetics."**

It demonstrates how to use **Twin Delayed DDPG (TD3)** with a custom **Bio-Prosthetic Wrapper** to solve the "Safety Gap" in reinforcement learning for biomechanics.

#### Key Features
* **BioProstheticWrapper:** A custom Gym wrapper that penalizes high-frequency torque changes (jerk).
* **Comparison:** Benchmarks TD3 (Conservative) against DDPG (Optimistic).
* **Analysis:** Includes scripts to visualize actuator voltage, variance, and clinical safety metrics.

## Installation
1. Install the dependencies:
   ```bash
   pip install -r requirements.txt

## Usage
Train the agents:

```bash
python train.py
This trains both DDPG and TD3 agents for 80,000 steps.
```

Run the Clinical Safety Analysis (Boxplots):

```bash
python visualize.py
```

Analyze Control Signals (Voltage Graphs):

```bash
python visualize_advanced.py
```

## Files Structure

### Core Logic
* `bio_env.py`: Custom gym wrapper that implements the biomechanical constraints (jerk penalty).
* `train.py`: Main training script for TD3 (Conservative) vs DDPG (Standard).
* `train_ablation.py`: Trains the "Raw" TD3 baseline (no bio-constraints) for ablation studies.

### Reproducing Figures
* `train_logger.py` & `plot_curve.py`: Generate the training logs and plot the **Learning Efficiency Curve (Fig 3)**.
* `visualize.py`: Runs clinical trials to generate the **Safety/Jerk Boxplots (Fig 4)**.
* `visualize_advanced.py`: plots the **Actuator Voltage Control Signals (Fig 6)**.

### Diagram Generation
* `Twin_Critic_Safety_Brake_diagram.py`: Python script used to generate the **Twin Critic architecture diagram (Fig 1)**.

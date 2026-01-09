import gymnasium as gym
import numpy as np

class BioProstheticWrapper(gym.Wrapper):
    """
    Wraps the standard Pendulum to act like a prosthetic joint.
    """
    def __init__(self, env, jerk_penalty_weight=0.1, randomize_patient=False):
        super().__init__(env)
        self.jerk_weight = jerk_penalty_weight
        self.randomize = randomize_patient
        # FIX 1: Ensure action is the correct shape and type (float32)
        self.last_action = np.zeros(self.env.action_space.shape, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Reset last_action to zero at the start of new episode
        self.last_action = np.zeros(self.env.action_space.shape, dtype=np.float32)
        
        # SIMULATE PATIENT VARIABILITY
        # We modify physics parameters *after* reset but before the next step.
        if self.randomize:
            new_m = np.random.uniform(0.8, 1.2) 
            new_l = np.random.uniform(0.8, 1.2) 
            self.env.unwrapped.m = new_m
            self.env.unwrapped.l = new_l
            
        return obs, info

    def step(self, action):
        # FIX 2: Explicitly define "Control Jerk" (squared change in input)
        # This represents the "smoothness" of the motor command.
        jerk = np.sum((action - self.last_action)**2)
        self.last_action = action
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Penalize reward
        modified_reward = reward - (self.jerk_weight * jerk)
        
        # Log jerk for later analysis
        info['jerk'] = jerk
        
        return obs, modified_reward, terminated, truncated, info

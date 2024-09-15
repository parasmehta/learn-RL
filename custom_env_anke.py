"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled
import torch
from gym.envs.classic_control.cartpole import CartPoleEnv

class LearnedEnv(CartPoleEnv):
    def __init__(self, render_mode: Optional[str] = None):
        super(LearnedEnv, self).__init__(render_mode)
        self.current_step = 0
        self.max_steps = 500
        self.model = torch.load("model.pt")

    
    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        # Combine state and action into one tensor
        state_action = torch.tensor(
            np.append(self.state, action), dtype=torch.float32)

        # Predict the next state using the neural network model
        next_state = self.model(state_action).detach().numpy()

    
        # Update the state
        self.state = next_state
        x, x_dot, theta, theta_dot = self.state
    
      
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            # Pole just fell!
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            if self.steps_beyond_terminated == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

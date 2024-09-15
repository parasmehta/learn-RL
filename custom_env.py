import gymnasium as gym
from gymnasium.envs.classic_control import CartPoleEnv

# Custom environment based on CartPole-v1


class CustomCartPoleEnv(CartPoleEnv):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self):
        super(CustomCartPoleEnv, self).__init__()

    def reset(self):
        # Reset the environment's state to a random state within the observation space
        self.state = self.observation_space.sample()
        self.current_step = 0
        return self.state

    def step(self, action):
        # Combine state and action into one tensor
        state_action = torch.tensor(
            np.append(self.state, action), dtype=torch.float32)

        # Predict the next state using the neural network model
        next_state = self.model(state_action).detach().numpy()

        # Update the state
        self.state = next_state

        # Calculate reward: same as CartPole-v1 (reward = 1 for every step)
        reward = 1.0

        # Determine if the episode is done
        done = bool(
            self.state[0] < -2.4  # Cart position
            or self.state[0] > 2.4
            # Pole angle in radians (approx. 12 degrees)
            or self.state[2] < -0.2095
            or self.state[2] > 0.2095
            or self.current_step >= self.max_steps  # Max steps limit
        )

        self.current_step += 1

        return self.state, reward, terminated, truncated, {}

    def render(self, mode="human"):
        # For now, we will skip rendering, but you could add visualization
        pass

    def close(self):
        pass

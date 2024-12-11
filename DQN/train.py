from PIL import Image
from collections import deque, namedtuple
import datetime
from itertools import count
import os
import shutil
import gymnasium as gym
from torchvision.utils import torch
import torch.nn as nn
import random
import envs
import numpy as np
import tensorflow as tf

from model import DQN

# Define the structure of a Transition using a namedtuple for memory storage
Transition = namedtuple(
    "Transition", ("state", "action", "next_state", "reward", "terminated")
)

# Experience Replay Memory Class to store and sample transitions
class MemoryReplay(object):
    def __init__(self, capacity):
        # Initialize a deque to store the memory of transitions with a fixed maximum size
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        # Append new transition to memory
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        # Sample a batch of transitions from memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        # Return the size of the memory buffer
        return len(self.memory)


class Trainer:
    def __init__(
        self,
        env: envs.Wrapper,  # Environment to interact with
        policy_net: DQN,  # The model that will be trained (policy network)
        target_net: DQN,  # The model used for computing target Q-values (target network)
        n_episodes=5000,  # Number of episodes to train for
        lr=1e-3,  # Learning rate for optimizer
        batch_size=32,  # Size of each mini-batch for optimization
        replay_size=10000,  # Capacity of the replay buffer
        learning_start=10000,  # Number of frames before learning starts
        target_update_freq=1000,  # Frequency of target network updates
        optimize_freq=3,  # Frequency of policy network optimization
        gamma=0.99,  # Discount factor for future rewards
        eps_start=1.0,  # Initial exploration rate
        eps_end=0.01,  # Final exploration rate
        eps_decay=10000,  # Decay rate for exploration
        window_size=5,  # Window size for calculating running averages of rewards
    ):
        # Store environment, networks, and hyperparameters
        self.env = env
        self.memory_replay = MemoryReplay(replay_size)

        # Set device for training (GPU or CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # Initialize the Q and Target networks
        self.policy_net = policy_net.to(self.device)
        self.target_net = target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        # Track number of training steps
        self.n_steps = 0

        # Store training hyperparameters
        self.n_episodes = n_episodes
        self.batch_size = batch_size

        # Set optimizer (AdamW optimizer with weight decay)
        self.optimizer = torch.optim.AdamW(
            self.policy_net.parameters(), lr=lr, amsgrad=True
        )

        # Store learning-related parameters
        self.learning_start = learning_start
        self.target_update_freq = target_update_freq
        self.optimize_freq = optimize_freq

        self.gamma = gamma

        # Exploration strategy (epsilon-greedy)
        self._get_eps = lambda n_steps: eps_end + (eps_start - eps_end) * np.exp(
            -1.0 * n_steps / eps_decay
        )

        # Set up the directory for saving training results
        folder_name = datetime.datetime.now().strftime("%y-%m-%d-%H-%M")
        folder_path = os.path.join("results", folder_name)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)  # Remove any existing folder
        os.makedirs(folder_path)  # Create new folder
        self.folder_path = folder_path
        self.loss_per_episode = []  # List to store losses per episode
        self.window_size = window_size
        self.rewards = deque(maxlen=window_size)  # Store rewards to calculate running averages


    def _select_action(self, state: torch.Tensor) -> torch.Tensor:
        """Select the next action given the current state following the eps-greedy policy"""
        eps = self._get_eps(self.n_steps)

        if random.random() > eps:
            # If we exploit, use the policy network to select the best action
            with torch.no_grad():
                return self.policy_net(state.unsqueeze(0)).max(dim=1)[1][0]
        else:
            # If we explore, sample a random action from the action space
            return torch.tensor(self.env.action_space.sample(), device=self.device)

    def _optimize(self):
        """Perform a step of optimization using a batch from the replay buffer"""
        # Sample a batch of transitions from the memory replay
        transitions = self.memory_replay.sample(self.batch_size)

        # Convert batch-array of Transitions to a Transition of batch-arrays
        batch = Transition(*zip(*transitions))

        # Stack the states, actions, next_states, rewards, and terminated flags
        state_batch = torch.stack(batch.state)
        action_batch = torch.stack(batch.action)
        next_state_batch = torch.stack(batch.next_state)
        reward_batch = torch.stack(batch.reward)
        terminated_batch = torch.tensor(
            batch.terminated, device=self.device, dtype=torch.float
        )

        # Compute batch "Q(s, a)"
        # The model returns "Q(s)", then we select the columns of actions taken.
        Q_values = (
            self.policy_net(state_batch)
            .gather(1, action_batch.unsqueeze(-1))
            .squeeze(-1)
        )

        # Compute batch "max_{a'} Q(s', a')"
        with torch.no_grad():
            next_Q_values = self.target_net(next_state_batch).max(1)[0]
        expected_Q_values = reward_batch + (1.0 - terminated_batch) * self.gamma * next_Q_values

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(Q_values, expected_Q_values)
        self.loss_per_episode.append(loss.item())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        file_writer = tf.summary.create_file_writer(f"dino_chrome_tensorboard/{current_time}")
        for episode_i in range(self.n_episodes):
            # Reset the environment to the initial state
            state, _ = self.env.reset()
            state = torch.tensor(state, device=self.device)
            self.loss_per_episode = []  # Reset list of losses for the episode

            total_reward = 0.0 # Initialize total reward count for the episode
            num_frames = 0  # Initialize frame count for the episode

            for t in count():
                self.n_steps += 1

                action = self._select_action(state)

                next_state, reward, terminated, *_ = self.env.step(envs.Action(action.item()))
                next_state = torch.tensor(next_state, device=self.device)

                total_reward += float(reward)
                num_frames += 1  # Increment the frame count each timestep

                self.memory_replay.push(state,action,next_state,torch.tensor(reward, device=self.device),terminated)

                # Synchronize the target network with the policy network
                if (self.n_steps > self.learning_start and self.n_steps % self.target_update_freq == 0):
                    print("sinchronize target network ", t)
                    self.target_net.load_state_dict(self.policy_net.state_dict())

                # Optimize the policy network
                if (self.n_steps > self.learning_start and self.n_steps % self.optimize_freq == 0):
                    self._optimize()

                # print(terminated)
                if terminated:
                    print(f"{episode_i} episode, done in {t+1} steps, total reward: {total_reward}")
                    break
                else:
                    state = next_state

            # Add total reward of this episode to the rewards deque (to calculate running average)
            self.rewards.append(total_reward)

            # Compute the running reward (average of rewards in the window)
            running_reward = np.mean(self.rewards)

            # Compute the average loss over episodes
            mean_loss_per_episode = np.mean(self.loss_per_episode)

            # Log the results to TensorBoard
            with file_writer.as_default():
                tf.summary.scalar('total_reward', total_reward, step = episode_i)
                tf.summary.scalar('num_frames', num_frames, step = episode_i)
                tf.summary.scalar('loss', mean_loss_per_episode, step = episode_i)
                tf.summary.scalar('running_reward', running_reward, step=episode_i)  # Log running reward

            # Save results periodically
            if episode_i % 50 == 0:
                self.save_obs_result(episode_i, self.env.frames)
                self.save_model_weights(episode_i)

        # Close the environment after training
        self.env.close()

    def save_obs_result(self, episode_i: int, obs_arr: list[np.ndarray]):
        # Save the observation frames as a GIF file
        frames = [Image.fromarray(obs, "RGB") for obs in obs_arr]
        file_path = os.path.join(self.folder_path, f"episode-{episode_i}.gif")

        frames[0].save(
            file_path,
            save_all=True,
            append_images=frames[1:],
            optimize=True,
            duration=100,
            loop=0,
        )

    def save_model_weights(self, episode_i: int):
        # Save the current model weights to a file
        file_path = os.path.join(self.folder_path, f"model-{episode_i}.pth")
        torch.save(self.policy_net, file_path)


if __name__ == "__main__":
    # Create and wrap the environment
    env = gym.make("Env-v0", render_mode="rgb_array", game_mode="train")
    env = envs.Wrapper(env, k=4)

    # Define the DQN networks
    obs_space = env.observation_space.shape
    assert obs_space is not None
    in_channels = obs_space[0]
    out_channels = env.action_space.n

    policy_net = DQN(in_channels, out_channels)
    target_net = DQN(in_channels, out_channels)

    # Initialize the trainer and start training
    trainer = Trainer(env, policy_net, target_net)
    trainer.train()

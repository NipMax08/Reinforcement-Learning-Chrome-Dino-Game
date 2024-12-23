import gymnasium as gym
import pygame
import torch
import envs
import argparse
import matplotlib.pyplot as plt
import numpy as np

from model import DQN


def human_play():
    env = gym.make("Env-v0", render_mode="human")
    obs, _ = env.reset()

    total_reward = 0.0
    n_frames = 0
    while True:
        n_frames += 1
        userInput = pygame.key.get_pressed()
        action = envs.Action.STAND
        if userInput[pygame.K_UP] or userInput[pygame.K_SPACE]:
            action = envs.Action.JUMP
        elif userInput[pygame.K_DOWN]:
            action = envs.Action.DUCK

        obs, reward, terminated, _, _ = env.step(action)

        total_reward += float(reward)
        if terminated:
            break

    print(f"Total reward: {total_reward}, number of frames: {n_frames}")

    env.close()

    # Show image of the last frame
    plt.imshow(obs)
    plt.show()


def play_with_model(
    env: envs.Wrapper,
    policy_net: DQN,
    device: torch.device,
    seed: int | None = None,
) -> float:
    if seed is not None:
        state, _ = env.reset(seed=seed)
    else:
        state, _ = env.reset()

    state = torch.tensor(state, device=device)

    total_reward = 0.0
    while True:
        action = policy_net(state.unsqueeze(0)).max(dim=1)[1][0]

        state, reward, terminated, _, _ = env.step(action)
        state = torch.tensor(state, device=device)

        total_reward += float(reward)
        if terminated:
            break

    return total_reward


def ai_play(model_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("True")
    else:
        print("False")
    policy_net = torch.load(model_path, map_location=device).to(device)
    policy_net.eval()

    env = gym.make("Env-v0", render_mode="human")
    env = envs.Wrapper(env)

    total_reward = play_with_model(env, policy_net, device)

    print(f"Total reward: {total_reward}, number of frames: {len(env.frames)}")

    env.close()

    # Show image of the last frame
    plt.imshow(env.frames[-1])
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("type", choices=["human", "ai"])
    parser.add_argument("-m", "--model_path")

    args = parser.parse_args()
    if args.type == "human":
        human_play()
    else:
        ai_play(args.model_path)

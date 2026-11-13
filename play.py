"""
Load a trained DQN model and watch it play the Atari game in real time.

Example:
    uv run play.py                                         # uses dqn_model.pth
    uv run play.py --model models/alice/exp_03/dqn_model.pth --episodes 2
"""

import argparse

import numpy as np
import torch

# Reuse the exact same environment setup and network architecture as training
from train import QNetwork, make_env


def main():
    parser = argparse.ArgumentParser(description="Watch a trained DQN agent play")
    parser.add_argument("--model", default="dqn_model.pth", help="path to saved weights")
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    # Same environment as training, but rendered in a window
    env = make_env(render_mode="human")

    # Rebuild the network and load the trained weights
    q_net = QNetwork(env.action_space.n)
    q_net.load_state_dict(torch.load(args.model, map_location="cpu"))
    q_net.eval()  # evaluation mode: no training

    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        total_reward, steps, done = 0.0, 0, False
        while not done:
            # Greedy policy: always pick the action with the highest Q-value
            with torch.no_grad():
                q_values = q_net(torch.as_tensor(np.array(state)).unsqueeze(0))
            action = q_values.argmax(dim=1).item()

            state, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        print(f"Episode {episode}: reward = {total_reward}, steps = {steps}")

    env.close()


if __name__ == "__main__":
    main()

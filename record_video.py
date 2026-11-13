"""
Record one game played by the trained agent and save it as an MP4 video.

Example:
    uv run record_video.py                       # uses dqn_model.pth -> gameplay.mp4
"""

import argparse

import cv2
import numpy as np
import torch

from train import QNetwork, make_env


def main():
    parser = argparse.ArgumentParser(description="Record the trained agent playing")
    parser.add_argument("--model", default="dqn_model.pth")
    parser.add_argument("--output", default="gameplay.mp4")
    args = parser.parse_args()

    env = make_env(render_mode="rgb_array")
    q_net = QNetwork(env.action_space.n)
    q_net.load_state_dict(torch.load(args.model, map_location="cpu"))
    q_net.eval()

    # Each env step covers 4 game frames (frame skip), so 15 steps per second = real speed
    height, width, scale = 210, 160, 3  # Atari screen size, enlarged 3x for the video
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"avc1"), 15, (width * scale, height * scale))

    state, _ = env.reset()
    total_reward, done = 0.0, False
    while not done:
        with torch.no_grad():
            action = q_net(torch.as_tensor(np.array(state)).unsqueeze(0)).argmax(dim=1).item()
        state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

        frame = env.render()  # RGB image of the screen
        frame = cv2.resize(frame, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    writer.release()
    env.close()
    print(f"Saved {args.output} (final score {total_reward:+.0f})")


if __name__ == "__main__":
    main()

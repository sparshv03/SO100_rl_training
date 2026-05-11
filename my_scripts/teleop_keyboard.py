"""
Keyboard teleop for the SO-100 gym environment.

Controls
--------
Joint selection:  1-6  (selects which joint to move)
Move joint:       LEFT / RIGHT arrow keys  (or A / D)
Gripper:          O = open,  C = close
Reset episode:    R
Quit:             Q or ESC

Currently selected joint is shown in the HUD.
"""

import os
os.environ.setdefault("MUJOCO_GL", "osmesa")  # osmesa = software renderer, doesn't conflict with pygame's GL context

import sys
import numpy as np
import gymnasium as gym
import gym_so100  # registers the envs
import pygame

ENV_ID = "gym_so100/SO100TouchCube-v0"
JOINT_NAMES = [
    "waist",
    "shoulder",
    "elbow",
    "forearm_roll",
    "wrist_rotate",
    "gripper",
]
STEP_SIZE = 0.05          # action delta per keypress (already in [-1,1] space)
GRIPPER_STEP = 0.1
FPS = 30


def make_env():
    return gym.make(ENV_ID, render_mode="rgb_array",
                    observation_width=640, observation_height=480,
                    visualization_width=640, visualization_height=480)


def draw_hud(surface, font, joint_idx, action, reward, step):
    lines = [
        f"Step: {step}   Reward: {reward:.4f}",
        f"Selected joint [{joint_idx+1}]: {JOINT_NAMES[joint_idx]}",
        f"Action: {np.round(action, 3)}",
        "",
        "Keys: 1-6=select joint  LEFT/RIGHT or A/D=move",
        "      O=open gripper  C=close gripper",
        "      R=reset  Q/ESC=quit",
    ]
    y = 8
    for line in lines:
        surf = font.render(line, True, (255, 255, 0))
        surface.blit(surf, (8, y))
        y += font.get_height() + 2


def get_frame(env):
    """Render from front_close — best view of arm and cube."""
    return env.unwrapped._env.physics.render(height=480, width=640, camera_id="front_close")


def run():
    env = make_env()
    obs, _ = env.reset(seed=0)

    pygame.init()
    frame = get_frame(env)                 # (H, W, 3)
    H, W = frame.shape[:2]
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("SO-100 Keyboard Teleop")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("monospace", 14)

    action = np.zeros(6, dtype=np.float32)
    joint_idx = 0
    total_reward = 0.0
    step = 0
    running = True

    print("Teleop started. Use keys shown in the window to control the robot.")

    while running:
        delta = np.zeros(6, dtype=np.float32)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                k = event.key

                # Quit
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

                # Reset
                elif k == pygame.K_r:
                    obs, _ = env.reset()
                    action = np.zeros(6, dtype=np.float32)
                    total_reward = 0.0
                    step = 0
                    print("Episode reset.")

                # Joint selection  1-6
                elif pygame.K_1 <= k <= pygame.K_6:
                    joint_idx = k - pygame.K_1
                    print(f"Selected joint {joint_idx+1}: {JOINT_NAMES[joint_idx]}")

                # Move selected joint
                elif k in (pygame.K_LEFT, pygame.K_a):
                    delta[joint_idx] = -STEP_SIZE
                elif k in (pygame.K_RIGHT, pygame.K_d):
                    delta[joint_idx] = +STEP_SIZE

                # Gripper shortcuts
                elif k == pygame.K_o:
                    delta[5] = +GRIPPER_STEP   # open
                elif k == pygame.K_c:
                    delta[5] = -GRIPPER_STEP   # close

        # Also handle held keys for smooth control
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            delta[joint_idx] = -STEP_SIZE
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            delta[joint_idx] = +STEP_SIZE
        if keys[pygame.K_o]:
            delta[5] = +GRIPPER_STEP
        if keys[pygame.K_c]:
            delta[5] = -GRIPPER_STEP

        if np.any(delta != 0):
            action = np.clip(action + delta, -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1

            if terminated:
                print(f"SUCCESS! Total reward: {total_reward:.2f} in {step} steps")
                obs, _ = env.reset()
                action = np.zeros(6, dtype=np.float32)
                total_reward = 0.0
                step = 0
            elif truncated:
                print(f"Episode truncated. Total reward: {total_reward:.2f}")
                obs, _ = env.reset()
                action = np.zeros(6, dtype=np.float32)
                total_reward = 0.0
                step = 0

        # Render
        frame = get_frame(env)
        surf = pygame.image.frombuffer(frame.tobytes(), (W, H), "RGB")
        screen.blit(surf, (0, 0))
        draw_hud(screen, font, joint_idx, action, total_reward, step)
        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()
    print("Teleop ended.")


if __name__ == "__main__":
    run()

"""
Reward Probe — interactively test the reward function.

Run this to understand what the reward looks like before training.
You control the robot with keys and watch reward values in real time.

Controls: same as teleop_keyboard.py
  1-6 = select joint
  LEFT/RIGHT or A/D = move joint
  O/C = open/close gripper
  R = reset episode
  Q/ESC = quit

Watch the terminal — it prints reward details every step.
"""

import os
os.environ.setdefault("MUJOCO_GL", "osmesa")

import numpy as np
import gymnasium as gym
import gym_so100
import pygame

# ── CHANGE THIS to test different environments ──────────────────────────────
ENV_ID   = "gym_so100/SO100TouchCube-v0"
OBS_TYPE = "so100_state"   # state obs so reset is fast (no pixel overhead)
# ────────────────────────────────────────────────────────────────────────────

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_rotate", "gripper"]
STEP_SIZE   = 0.05
FPS         = 30


def get_frame(env):
    return env.unwrapped._env.physics.render(height=480, width=640, camera_id="front_close")


def get_reward_details(env):
    """Pull raw numbers straight from physics so we can inspect them."""
    physics  = env.unwrapped._env.physics
    task     = env.unwrapped._env.task

    # gripper tip position
    ee_id  = physics.model.site("ee_site").id
    ee_pos = physics.data.site_xpos[ee_id].copy()

    # cube position
    cube_id  = physics.model.site("cube_site").id
    cube_pos = physics.data.site_xpos[cube_id].copy()

    dist = np.linalg.norm(ee_pos - cube_pos)

    # contact detection
    CUBE_GEOM   = "red_box"
    FINGER_GEOMS = {f"fixed_jaw_pad_{i}" for i in range(1, 5)} | \
                   {f"moving_jaw_pad_{i}" for i in range(1, 5)}

    contacts = []
    for i in range(physics.data.ncon):
        g1 = physics.model.id2name(physics.data.contact[i].geom1, "geom")
        g2 = physics.model.id2name(physics.data.contact[i].geom2, "geom")
        contacts.append((g1, g2))

    touching = any(
        (g1 in FINGER_GEOMS and g2 == CUBE_GEOM) or
        (g2 in FINGER_GEOMS and g1 == CUBE_GEOM)
        for g1, g2 in contacts
    )

    return {
        "ee_pos":      np.round(ee_pos, 3),
        "cube_pos":    np.round(cube_pos, 3),
        "dist":        round(float(dist), 4),
        "touching":    touching,
    }


def draw_hud(surface, font, joint_idx, action, episode_reward, step, reward, details):
    lines = [
        f"─── REWARD PROBE ───────────────────────",
        f"Step: {step}   Episode reward so far: {episode_reward:.3f}",
        f"",
        f"  Last step reward:    {reward:+.4f}",
        f"  EE→Cube distance:   {details['dist']:.4f} m",
        f"  Gripper touching:   {'YES ✓' if details['touching'] else 'no'}",
        f"  EE position:        {details['ee_pos']}",
        f"  Cube position:      {details['cube_pos']}",
        f"",
        f"─── CONTROL ────────────────────────────",
        f"  Joint [{joint_idx+1}]: {JOINT_NAMES[joint_idx]}",
        f"  Action: {np.round(action, 2)}",
        f"",
        f"  Keys: 1-6=joint  ←/→=move  O/C=gripper",
        f"        R=reset  Q=quit",
    ]
    y = 8
    for line in lines:
        color = (100, 255, 100) if "YES" in line else \
                (255, 80,  80)  if "no"  in line else \
                (255, 255,  0)
        surf = font.render(line, True, color)
        surface.blit(surf, (8, y))
        y += font.get_height() + 2


def print_reward_breakdown(step, reward, details, action):
    """Print a detailed reward breakdown to the terminal each step."""
    d = details["dist"]
    t = details["touching"]

    # Reconstruct what the reward function is doing
    shape = 0.0
    if d < 0.7:  shape = max(shape, 0.1 * (1 - d / 0.7))
    if d < 0.5:  shape = max(shape, 0.2 * (1 - d / 0.5))
    if d < 0.3:  shape = max(shape, 0.5 * (1 - d / 0.3))
    if d < 0.1:  shape = max(shape, 1.0 * (1 - d / 0.1))
    if d < 0.05: shape = max(shape, 2.0 * (1 - d / 0.05))

    print(
        f"step={step:3d} | "
        f"dist={d:.3f}m | "
        f"touch={'Y' if t else 'N'} | "
        f"shape={shape:+.3f} | "
        f"touch_bonus={1.0 if t else 0.0:+.1f} | "
        f"step_penalty=-0.20 | "
        f"reward={reward:+.4f}"
    )


def run():
    env = gym.make(ENV_ID, obs_type=OBS_TYPE, render_mode="rgb_array",
                   observation_width=640, observation_height=480)
    obs, _ = env.reset(seed=42)

    pygame.init()
    frame  = get_frame(env)
    H, W   = frame.shape[:2]
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Reward Probe")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("monospace", 13)

    action         = np.zeros(6, dtype=np.float32)
    joint_idx      = 0
    episode_reward = 0.0
    step           = 0
    last_reward    = 0.0
    last_details   = get_reward_details(env)
    running        = True

    print("\nReward Probe started. Move the robot and watch reward values.\n")
    print(f"{'step':>4}  {'dist':>7}  touch  {'shape':>7}  {'touch_bonus':>11}  {'penalty':>9}  {'reward':>8}")
    print("-" * 65)

    while running:
        delta = np.zeros(6, dtype=np.float32)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif k == pygame.K_r:
                    obs, _ = env.reset()
                    action = np.zeros(6, dtype=np.float32)
                    episode_reward = 0.0
                    step = 0
                    print("\n--- RESET ---\n")
                elif pygame.K_1 <= k <= pygame.K_6:
                    joint_idx = k - pygame.K_1
                elif k in (pygame.K_LEFT,  pygame.K_a): delta[joint_idx] = -STEP_SIZE
                elif k in (pygame.K_RIGHT, pygame.K_d): delta[joint_idx] = +STEP_SIZE
                elif k == pygame.K_o: delta[5] = +0.1
                elif k == pygame.K_c: delta[5] = -0.1

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]  or keys[pygame.K_a]: delta[joint_idx] = -STEP_SIZE
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: delta[joint_idx] = +STEP_SIZE
        if keys[pygame.K_o]: delta[5] = +0.1
        if keys[pygame.K_c]: delta[5] = -0.1

        if np.any(delta != 0):
            action = np.clip(action + delta, -1.0, 1.0)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step           += 1
            last_reward     = reward
            last_details    = get_reward_details(env)

            print_reward_breakdown(step, reward, last_details, action)

            if terminated:
                print(f"\n🎉 SUCCESS in {step} steps! Total reward: {episode_reward:.2f}\n")
                obs, _ = env.reset()
                action = np.zeros(6, dtype=np.float32)
                episode_reward = 0.0
                step = 0
            elif truncated:
                print(f"\nEpisode ended (300 steps). Total reward: {episode_reward:.2f}\n")
                obs, _ = env.reset()
                action = np.zeros(6, dtype=np.float32)
                episode_reward = 0.0
                step = 0

        frame = get_frame(env)
        surf  = pygame.image.frombuffer(frame.tobytes(), (W, H), "RGB")
        screen.blit(surf, (0, 0))
        draw_hud(screen, font, joint_idx, action, episode_reward, step, last_reward, last_details)
        pygame.display.flip()
        clock.tick(FPS)

    env.close()
    pygame.quit()
    print("\nProbe ended.")


if __name__ == "__main__":
    run()

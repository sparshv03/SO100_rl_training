"""
Visualize a trained SAC policy.

Usage:
  python my_scripts/visualize_policy.py
  python my_scripts/visualize_policy.py --model outputs/checkpoints/SO100TouchCube/sac_SO100TouchCube_150000_steps.zip
  python my_scripts/visualize_policy.py --save-video outputs/videos/policy.mp4
  python my_scripts/visualize_policy.py --episodes 5
"""

import os
os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import numpy as np
import gymnasium as gym
import gym_so100
import pygame
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv

DEFAULT_MODEL   = "outputs/checkpoints/SO100TouchCube/sac_SO100TouchCube_final.zip"
DEFAULT_VECNORM = "outputs/checkpoints/SO100TouchCube/sac_SO100TouchCube_final_vecnorm.pkl"
TASK_ID         = "gym_so100/SO100TouchCube-v0"
OBS_TYPE        = "so100_state"
FPS             = 30


def get_inner_env(vec_env):
    """Unwrap DummyVecEnv -> VecNormalize -> gymnasium env -> SO100Env."""
    # DummyVecEnv stores envs in .envs; VecNormalize wraps another VecEnv in .venv
    base = vec_env
    while hasattr(base, "venv"):
        base = base.venv
    return base.envs[0].unwrapped


def get_frame(vec_env):
    inner = get_inner_env(vec_env)
    return inner._env.physics.render(height=480, width=640, camera_id="front_close")


def get_dist_and_touch(vec_env):
    inner   = get_inner_env(vec_env)
    physics = inner._env.physics

    ee_id   = physics.model.site("ee_site").id
    cube_id = physics.model.site("cube_site").id
    ee_pos   = physics.data.site_xpos[ee_id]
    cube_pos = physics.data.site_xpos[cube_id]
    dist     = float(np.linalg.norm(ee_pos - cube_pos))

    CUBE = "red_box"
    PADS = {f"fixed_jaw_pad_{i}" for i in range(1, 5)} | \
           {f"moving_jaw_pad_{i}" for i in range(1, 5)}
    touching = any(
        (physics.model.id2name(physics.data.contact[i].geom1, "geom") in PADS and
         physics.model.id2name(physics.data.contact[i].geom2, "geom") == CUBE) or
        (physics.model.id2name(physics.data.contact[i].geom2, "geom") in PADS and
         physics.model.id2name(physics.data.contact[i].geom1, "geom") == CUBE)
        for i in range(physics.data.ncon)
    )
    return dist, touching


def draw_hud(surface, font, step, ep, reward, ep_reward, action, dist, touching, deterministic):
    lines = [
        f"AI POLICY  ({'DETERMINISTIC' if deterministic else 'STOCHASTIC'})",
        f"Episode: {ep}   Step: {step}",
        f"",
        f"  Step reward:   {reward:+.4f}",
        f"  Episode total: {ep_reward:+.4f}",
        f"  EE->Cube dist: {dist:.4f} m",
        f"  Touching:      {'YES' if touching else 'no'}",
        f"",
        f"  Action: {np.round(action, 2)}",
        f"",
        f"  D=deterministic  S=stochastic  R=reset  Q=quit",
    ]
    y = 8
    for line in lines:
        color = (100, 255, 100) if "YES" in line else \
                (255, 80,  80)  if line.strip().startswith("Touching") and "no" in line else \
                (255, 255, 0)
        surface.blit(font.render(line, True, color), (8, y))
        y += font.get_height() + 2


def build_vec_env(vecnorm_path):
    vec_env = DummyVecEnv([lambda: gym.make(TASK_ID, obs_type=OBS_TYPE,
                                            render_mode="rgb_array")])
    if os.path.exists(vecnorm_path):
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False
        print(f"Loaded VecNormalize: {vecnorm_path}")
    else:
        print("Warning: vecnorm file not found — obs won't be normalized")
    return vec_env


def run_window(model, vec_env, num_episodes, deterministic):
    pygame.init()
    frame  = get_frame(vec_env)
    H, W   = frame.shape[:2]
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("SAC Policy Viewer")
    font   = pygame.font.SysFont("monospace", 13)
    clock  = pygame.time.Clock()

    obs       = vec_env.reset()
    ep        = 1
    step      = 0
    ep_reward = 0.0
    last_r    = 0.0
    last_act  = np.zeros(6)
    running   = True

    print(f"\n{'='*50}")
    print(f" SAC agent — {'deterministic' if deterministic else 'stochastic'}")
    print(f"{'='*50}")
    print(f"{'ep':>3} {'step':>4} {'dist':>7} {'touch':>5} {'step_r':>8} {'ep_r':>8}")
    print("-" * 43)

    while running and ep <= num_episodes:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE): running = False
                elif k == pygame.K_r:
                    obs = vec_env.reset(); step = 0; ep_reward = 0.0
                elif k == pygame.K_d:
                    deterministic = True;  print("\n[DETERMINISTIC]")
                elif k == pygame.K_s:
                    deterministic = False; print("\n[STOCHASTIC]")

        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = vec_env.step(action)

        last_r     = float(rewards[0])
        ep_reward += last_r
        step      += 1
        last_act   = action[0]

        dist, touching = get_dist_and_touch(vec_env)
        print(f"{ep:>3} {step:>4} {dist:>7.3f} {'Y' if touching else 'N':>5} "
              f"{last_r:>+8.4f} {ep_reward:>+8.3f}")

        if dones[0]:
            success = infos[0].get("is_success", False)
            print(f"    Episode {ep}: reward={ep_reward:.3f} | "
                  f"{'SUCCESS ✓' if success else 'truncated'}\n")
            ep += 1; step = 0; ep_reward = 0.0

        frame = get_frame(vec_env)
        surf  = pygame.image.frombuffer(frame.tobytes(), (W, H), "RGB")
        screen.blit(surf, (0, 0))
        draw_hud(screen, font, step, ep, last_r, ep_reward, last_act,
                 dist, touching, deterministic)
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


def save_video(model, vec_env, num_episodes, deterministic, path):
    import imageio
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    frames = []
    obs    = vec_env.reset()
    ep     = 1
    step   = 0

    print(f"\nRecording {num_episodes} episode(s) → {path}")

    while ep <= num_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, rewards, dones, infos = vec_env.step(action)
        step += 1
        frames.append(get_frame(vec_env))

        if dones[0]:
            success = infos[0].get("is_success", False)
            print(f"  Episode {ep}: {step} steps | {'SUCCESS ✓' if success else 'truncated'}")
            ep += 1; step = 0

    imageio.mimsave(path, frames, fps=FPS)
    print(f"Saved → {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--vecnorm",    default=DEFAULT_VECNORM)
    parser.add_argument("--episodes",   type=int, default=3)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--save-video", default=None, metavar="PATH")
    args = parser.parse_args()

    vec_env = build_vec_env(args.vecnorm)
    model   = SAC.load(args.model, env=vec_env)
    print(f"Loaded model: {args.model}")

    if args.save_video:
        save_video(model, vec_env, args.episodes, not args.stochastic, args.save_video)
    else:
        run_window(model, vec_env, args.episodes, not args.stochastic)

    vec_env.close()


if __name__ == "__main__":
    main()

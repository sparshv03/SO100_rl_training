"""
SAC Training Script for SO-100 Robot Arm
=========================================

What this script does, step by step:
  1. Creates the gym environment
  2. Wraps it so SAC can handle it (vectorized + normalized observations)
  3. Builds a SAC agent (Actor + Critic neural networks)
  4. Runs the training loop:
       - Agent explores and collects experience
       - Experience is stored in a replay buffer
       - Networks are updated from sampled batches
  5. Saves checkpoints and logs to TensorBoard

Usage:
  python train/train_sac.py                         # train from scratch
  python train/train_sac.py --task SO100CubeToBin   # different task
  python train/train_sac.py --resume outputs/checkpoints/model_5000_steps.zip

Monitor with TensorBoard:
  tensorboard --logdir outputs/logs
"""

import os
os.environ.setdefault("MUJOCO_GL", "osmesa")

import argparse
import gymnasium as gym
import gym_so100  # registers the env IDs
import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.logger import configure

# ── CONFIG — change these to experiment ─────────────────────────────────────

TASKS = {
    "SO100TouchCube":       "gym_so100/SO100TouchCube-v0",        # dense reward, easiest
    "SO100TouchCubeSparse": "gym_so100/SO100TouchCubeSparse-v0",  # sparse reward, harder
    "SO100CubeToBin":       "gym_so100/SO100CubeToBin-v0",        # pick+place, hardest
}

# Observation type:
#   "so100_state"             → 12 numbers (fast, good for learning the basics)
#   "so100_pixels_agent_pos"  → camera image + joint angles (slow, more realistic)
OBS_TYPE = "so100_state"

# How many parallel environments to run during training
# More = faster data collection, but uses more RAM/CPU
NUM_ENVS = 4

# Total environment steps to train for
TOTAL_STEPS = 300_000

# Save a checkpoint every N steps
SAVE_FREQ = 10_000

# Where to save everything
OUTPUT_DIR = "outputs"

# ────────────────────────────────────────────────────────────────────────────


def make_env_fn(task_id, obs_type):
    """Returns a function that creates one environment instance."""
    def _init():
        env = gym.make(
            task_id,
            obs_type=obs_type,
            observation_width=64,    # small image if using pixels (faster)
            observation_height=48,
        )
        return env
    return _init


def build_training_env(task_id, obs_type, num_envs):
    """
    Build the vectorized + normalized training environment.

    VecEnv     = runs N envs in parallel, giving N experiences per step
    VecNormalize = keeps a running mean/std of observations and normalizes
                   them to ~zero mean, unit variance — makes neural net training stable
    """
    vec_env = make_vec_env(
        make_env_fn(task_id, obs_type),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv,   # each env in a separate process
    )
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,       # normalize observations
        norm_reward=False,   # do NOT normalize reward (keeps it interpretable)
        clip_obs=10.0,       # clip extreme obs values
    )
    return vec_env


def build_model(vec_env, log_dir):
    """
    Build the SAC agent.

    SAC has two main networks:
      - Actor  (policy): obs → action     — what to do
      - Critic (value):  obs + action → Q — how good is this action

    Both are MLPs (fully connected neural networks).
    """
    # Pick best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Training on: {device}")

    model = SAC(
        policy="MultiInputPolicy" if isinstance(vec_env.observation_space, gym.spaces.Dict)
               else "MlpPolicy",
        env=vec_env,

        # ── Learning rate ─────────────────────────────────────────────────
        # How big a gradient step to take each update.
        # Too high = unstable training. Too low = very slow.
        learning_rate=3e-4,

        # ── Replay buffer ─────────────────────────────────────────────────
        # How many past (obs, action, reward, next_obs) tuples to store.
        # Larger = more diverse experience, but more RAM.
        buffer_size=100_000,

        # ── Batch size ────────────────────────────────────────────────────
        # How many samples to draw from the buffer per update.
        batch_size=256,

        # ── Entropy coefficient ───────────────────────────────────────────
        # "auto" = SAC automatically tunes how much it encourages exploration.
        # Higher entropy = more random = more exploration.
        ent_coef="auto",

        # ── Learning starts ───────────────────────────────────────────────
        # Don't start training the networks until we have this many
        # random samples in the buffer. Pure random exploration first.
        learning_starts=1000,

        # ── Policy network architecture ───────────────────────────────────
        # Two hidden layers of 256 neurons each. Standard for robot tasks.
        policy_kwargs={"net_arch": [256, 256]},

        device=device,
        verbose=1,                   # print training progress
        tensorboard_log=log_dir,
    )

    # Connect TensorBoard logger
    logger = configure(log_dir, ["tensorboard", "stdout"])
    model.set_logger(logger)

    return model


class RewardLogCallback(BaseCallback):
    """
    Custom callback that logs extra info to TensorBoard every N steps.
    This is how you add your own metrics during training.
    """
    def __init__(self, log_freq=1000, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards = []

    def _on_step(self):
        # SB3 stores episode info in self.locals["infos"]
        for info in self.locals.get("infos", []):
            if "episode" in info:
                ep_reward = info["episode"]["r"]
                self.episode_rewards.append(ep_reward)
                self.logger.record("custom/episode_reward", ep_reward)

        if self.n_calls % self.log_freq == 0 and self.episode_rewards:
            mean_r = np.mean(self.episode_rewards[-20:])  # last 20 episodes
            self.logger.record("custom/mean_reward_last20", mean_r)
            print(f"  [step {self.num_timesteps}] mean reward (last 20 eps): {mean_r:.3f}")

        return True  # returning False would stop training


def train(task_name, obs_type, num_envs, total_steps, save_freq, resume_path=None):
    task_id = TASKS[task_name]

    # Create output directories
    log_dir        = os.path.join(OUTPUT_DIR, "logs", task_name)
    checkpoint_dir = os.path.join(OUTPUT_DIR, "checkpoints", task_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Task:      {task_id}")
    print(f"Obs type:  {obs_type}")
    print(f"Num envs:  {num_envs}")
    print(f"Steps:     {total_steps:,}")
    print(f"{'='*50}\n")

    # Build environment
    vec_env = build_training_env(task_id, obs_type, num_envs)

    # Build or load model
    if resume_path and os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}")
        model = SAC.load(resume_path, env=vec_env)
        # Load normalization stats if they exist alongside the checkpoint
        norm_path = resume_path.replace(".zip", "_vecnorm.pkl")
        if os.path.exists(norm_path):
            vec_env = VecNormalize.load(norm_path, vec_env)
            model.set_env(vec_env)
            print(f"Loaded VecNormalize stats: {norm_path}")
    else:
        model = build_model(vec_env, log_dir)

    # Build callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq // num_envs,   # per-env steps, not total
        save_path=checkpoint_dir,
        name_prefix=f"sac_{task_name}",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    reward_log_cb = RewardLogCallback(log_freq=500)

    # ── TRAINING LOOP ────────────────────────────────────────────────────────
    # This is where the magic happens.
    # SB3 handles the loop internally:
    #   1. Agent acts in all N envs → gets N (obs, action, reward, next_obs)
    #   2. Stores them in replay buffer
    #   3. Samples a batch → computes loss → updates Actor + Critic
    #   4. Repeats until total_steps reached
    print("Starting training...\n")
    model.learn(
        total_timesteps=total_steps,
        callback=[checkpoint_cb, reward_log_cb],
        reset_num_timesteps=resume_path is None,  # don't reset step counter on resume
        progress_bar=True,
    )

    # Save final model
    final_path = os.path.join(checkpoint_dir, f"sac_{task_name}_final")
    model.save(final_path)
    vec_env.save(final_path + "_vecnorm.pkl")
    print(f"\nTraining done. Final model saved to: {final_path}.zip")

    vec_env.close()
    return model


def evaluate(task_name, model_path, obs_type, num_episodes=10):
    """
    Run a trained model and print per-episode rewards.
    No learning — just watching what the policy learned to do.
    """
    task_id = TASKS[task_name]
    norm_path = model_path.replace(".zip", "_vecnorm.pkl")

    env = gym.make(task_id, obs_type=obs_type, render_mode="rgb_array")
    model = SAC.load(model_path)

    print(f"\nEvaluating {model_path} for {num_episodes} episodes...\n")
    total_rewards = []

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            # predict() returns (action, hidden_state)
            # deterministic=True = use mean of policy (no random sampling)
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            done = terminated or truncated

        total_rewards.append(ep_reward)
        print(f"  Episode {ep+1:2d}: reward = {ep_reward:.3f}")

    print(f"\nMean reward over {num_episodes} episodes: {np.mean(total_rewards):.3f}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",    default="SO100TouchCube",
                        choices=list(TASKS.keys()),
                        help="Which task to train on")
    parser.add_argument("--obs",     default=OBS_TYPE,
                        choices=["so100_state", "so100_pixels_agent_pos"],
                        help="Observation type")
    parser.add_argument("--envs",    type=int, default=NUM_ENVS,
                        help="Number of parallel environments")
    parser.add_argument("--steps",   type=int, default=TOTAL_STEPS,
                        help="Total training steps")
    parser.add_argument("--resume",  default=None,
                        help="Path to checkpoint .zip to resume from")
    parser.add_argument("--eval",    default=None,
                        help="Path to model .zip to evaluate (skips training)")
    args = parser.parse_args()

    if args.eval:
        evaluate(args.task, args.eval, args.obs)
    else:
        train(args.task, args.obs, args.envs, args.steps, SAVE_FREQ, args.resume)

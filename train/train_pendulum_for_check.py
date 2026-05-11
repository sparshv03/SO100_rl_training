import gymnasium as gym
from stable_baselines3 import SAC

env = gym.make("Pendulum-v1")

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./logs/"
)

model.learn(total_timesteps=10000)

model.save("models/sac_pendulum")
import numpy as np
import gym
import math

from matplotlib import pyplot as plt
from cartpole_env import CartPoleEnv
from stable_baselines.common.policies import MlpPolicy as Policy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO2

gym.envs.registry.register(
    id='ContinuousCartPole-v1',
    entry_point=CartPoleEnv,
    max_episode_steps=1000,
    reward_threshold=1000.0,
)
environment = 'ContinuousCartPole-v1'
env = DummyVecEnv([lambda: gym.make(environment)])
name = 'ppo2_cartpole_3'
model_folder = 'models/'
tensorboard_folder = 'tensorboard/'

def train(num_steps=int(1e6), new=False):
    # https://github.com/openai/baselines/issues/471
    env_train = DummyVecEnv([lambda: gym.make(environment)])
    if new:
        model = PPO2(Policy,
                     env_train,
                     verbose=1,
                     tensorboard_log=tensorboard_folder + name + '/',
                     n_steps=2048,
                     nminibatches=32,
                     lam=0.95,
                     gamma=0.99,
                     noptepochs=10,
                     ent_coef=0.0,
                     learning_rate=3e-4,
                     cliprange=0.2)
    else:
        model = PPO2.load(model_folder + name, env_train)
    model.learn(total_timesteps=num_steps)
    model.save(model_folder + name)
    del env_train

def evaluate(num_steps=1000):
    model = PPO2.load(model_folder + name, env)

    obs = env.reset()
    for i in range(num_steps):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if num_steps/3 < i < num_steps/3*2:
            obs[0][0] += rect(i/num_steps*np.pi*5)

def rect(x):
    s = math.sin(x)
    if s > 0:
        return 1.0
    elif s <= 0:
        return -1.0

if __name__ == '__main__':
    #train(int(1e6))
    evaluate(1000)

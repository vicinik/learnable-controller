""" Pendulum environment launcher.
Same principles as run_toy_env. See the docs for more details.

Authors: Vincent Francois-Lavet, David Taralla
"""

import sys
import logging
import numpy as np
import gym

from matplotlib import pyplot as plt
from stable_baselines.common.policies import MlpPolicy as Policy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import A2C, PPO2
from sbri_prod_env import SbriEnv
from mphy_regs import sbri_rx1_mphy_regs

regs = [sbri_rx1_mphy_regs[1],sbri_rx1_mphy_regs[2],sbri_rx1_mphy_regs[4],sbri_rx1_mphy_regs[6]]
env = SbriEnv(regs, 'Z:/nhaminge/smarti_bridge-sbri3/70_sscript_tc/release_regression_v2/last_eyescan.csv')

def trace():
    env.trace_full_system()
    env.save('sbri.env')

def train(algo, timesteps=100000, sim=True):
    env.load('sbri.env')
    env.set_simulation_enabled(sim)
    vec_env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run

    model = algo(Policy, vec_env, verbose=1)
    #model = algo.load('sbri', vec_env)
    model.learn(total_timesteps=timesteps)
    model.save('sbri')
    if not sim: env.save('sbri.env')

def evaluate(algo, timesteps=10, sim=True):
    env.load('sbri.env')
    env.set_simulation_enabled(sim)
    vec_env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
    model = algo.load('sbri', vec_env)

    x, y = env.get_system_func()
    plt.figure(figsize=(19, 10))
    plt.title('System Function')
    t = range(len(x))
    m = max([max(x), max(y)])
    plt.plot(t, [-k+m for k in x], 'b-')
    plt.plot(t, [-k+m for k in y], 'g-')

    obs = vec_env.reset()
    t = []
    a = []
    for i in range(timesteps):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        action = list(action[0])
        idx = env.get_action_index(action)
        t.append(idx)
        a.append(rewards)

    plt.plot(t, a, 'ro')
    plt.xlabel('Input'), plt.ylabel('Output')
    plt.legend(['Eye Diagram Width', 'Eye Diagram Height', 'Agent Actions'])
    plt.show()

if __name__ == '__main__':
    train(PPO2)
    evaluate(PPO2)
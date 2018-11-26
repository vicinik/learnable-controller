"""
This environment is a wrapper for communicating with the real system.

The inputs of the system are some parameters of the repeaters.
The outputs of the system are the width and the height of the
statistical eye diagram.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os, copy, time, json
from deer.base_classes import Environment
import mphy_regs
import gym
from gym import spaces
from gym.utils import seeding
import sbri_interface as sbri

class SbriEnv(gym.Env):

    def __init__(self, registers, eye_scan_file):
        self._registers = registers
        self._eye_scan_file = eye_scan_file
        inputs = np.array([.99 for i in range(len(registers))])
        outputs = np.array([500.0, 500.0])

        self.action_space = spaces.Box(-inputs, inputs, dtype=np.float32)
        self.observation_space = spaces.Box(-outputs, outputs, dtype=np.float32)

        self.best_reward = np.finfo(np.float32).min
        self.state = [0 for i in range(len(outputs))]
        self.reset_actions = [0 for i in range(len(inputs))]

        self.seed()
        
        self._system_trace = { 'x': {}, 'y': {}}
        self._sim_enabled = False

    def set_simulation_enabled(self, enabled=True):
        num_permutations = 1
        for reg in self._registers:
            num_permutations *= len(reg.mphy_reg.val_range)
        num_traced = len(self._system_trace['x'])
        if enabled and num_traced < num_permutations:
            raise RuntimeError('The system has not been traced fully yet, therefore a simulation is not possible.'
            ' Call "trace_full_system()" to trace it or "load()" to load a previously traced system.')
        self._sim_enabled = enabled

    def trace_full_system(self, num_values=1):
        print('The system will now be traced {} time(s). This might take a long time depending on the number of registers...'.format(num_values))
        for i in range(num_values):
            self._trace_full_system_rec()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        if not isinstance(action, (list, np.ndarray)):
            raise TypeError('There must be a list of actions, but action is {}'.format(action))
        elif len(self._registers) != len(action):
            raise RuntimeError('Number of actions ({}) does not match number of registers ({})'.format(len(action), len(self._registers)))
        
        discrete_actions = self.discretize_actions(action)
        #print('Current actions:', discrete_actions)

        x = 0.0
        y = 0.0

        if self._sim_enabled:
            x, y = self._get_traced_system_response(discrete_actions)
        else:
            x, y = self._get_real_system_response(discrete_actions)

        #print('Avg. X: {}, Avg. Y: {}'.format(x, y))

        reward = -(x + y) + 10.0
        if reward > self.best_reward:
            self.best_reward = reward
            self.reset_actions = action
        #print('Reward: {} | Best reward: {}'.format(reward, self.best_reward))

        return np.array([x, y]), reward, False, {}
    
    def reset(self):
        print('Reset environment')
        state, reward, done, _ = self.step(self.reset_actions)
        return state

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def discretize_actions(self, action):
        discrete_actions = []
        for i in range(len(action)):
            action_range = len(self._registers[i].mphy_reg.val_range)
            act = int((action[i]+1.0)*(action_range/2))
            if act > action_range-1: act = action_range-1
            elif act < 0: act = 0
            discrete_actions.append(act)
        return discrete_actions

    def get_system_func(self):
        keys_sorted_x = sorted(self._system_trace['x'].items(), key=lambda kv: kv[0])
        keys_sorted_y = sorted(self._system_trace['y'].items(), key=lambda kv: kv[0])
        x = []
        y = []
        for i, key_val in enumerate(keys_sorted_x):
            x.append(self._avg(key_val[1]))
        for i, key_val in enumerate(keys_sorted_y):
            y.append(self._avg(key_val[1]))
        return x, y

    def save(self, file_name):
        sys_str = json.dumps(self._system_trace)
        with open(file_name, 'w') as f:
            f.write(sys_str)

    def load(self, file_name):
        with open(file_name, 'r') as f:
            sys_str = f.read()
            self._system_trace = json.loads(sys_str)

    def get_action_index(self, action):
        discrete_actions = self.discretize_actions(action)
        key = ''.join([str(a) for a in discrete_actions])
        keys_sorted_x = sorted(self._system_trace['x'].items(), key=lambda kv: kv[0])
        for i, key_val in enumerate(keys_sorted_x):
            if key_val[0] == key: return i
        return -1

    def _get_real_system_response(self, discrete_actions):
        for i, reg in enumerate(self._registers):
            cur_val = sbri.get_mphy_reg_val(reg)
            new_val = reg.get_value_decorated(cur_val, discrete_actions[i])
            sbri.set_mphy_reg_val(reg, new_val)

        sbri.generate_eyescan(16, 4)
        x_values, y_values = sbri.parse_eyediagram_file(self._eye_scan_file)
        x = self._avg(x_values)*100
        y = self._avg(y_values)*100
        
        key = ''.join([str(a) for a in discrete_actions])
        if not (key in self._system_trace['x']):
            self._system_trace['x'][key] = []
            self._system_trace['y'][key] = []
        self._system_trace['x'][key].append(x)
        self._system_trace['y'][key].append(y)

        return x, y

    def _get_traced_system_response(self, discrete_actions):
        key = ''.join([str(a) for a in discrete_actions])
        x = self._avg(self._system_trace['x'][key])
        y = self._avg(self._system_trace['y'][key])

        return x, y

    def _trace_full_system_rec(self, idx=0, actions=None):
        if actions is None: actions = [0 for i in range(len(self._registers))]
        reg = self._registers[idx].mphy_reg
        for i in reg.val_range:
            actions[idx] = i
            if idx == len(self._registers) - 1:
                x, y = self._get_real_system_response(actions)
                print('Traced action {}: (x={}, y={})'.format(actions, x, y))
            else:
                self._trace_full_system_rec(idx+1, actions)

    def _avg(self, vals):
        return float(sum(vals)) / max(len(vals), 1)

class MyEnv(Environment):
    def __init__(self, registers, eye_scan_file):
        """ Initialize environment.
        """
        self._env = SbriEnv(registers, eye_scan_file)
        self._is_terminal = False
        self._input_dim = [(1,), (1,)]
        self._n_actions = [(-.99, .99) for i in range(len(registers))]

    def act(self, action):
        self._last_observation, reward, self._is_terminal, _ = self._env.step(action)

        #time.sleep(1)
        return reward
                
    def reset(self, mode=0):
        self._last_observation = self._env.reset()
        self._is_terminal = False

        return self._last_observation
                
    def inTerminalState(self):
        """Tell whether the environment reached a terminal state after the last transition (i.e. the last transition 
        that occured was terminal).
        """
        return self._is_terminal
        
    def summarizePerformance(self, test_data_set):
        """ This function is called at every PERIOD_BTW_SUMMARY_PERFS.

        Arguments:
            [test_data_set] Simulation data returned by the agent.
        """
        print('Called summarizePerformance')

    def inputDimensions(self):
        return self._input_dim

    def nActions(self):
        # The environment allows two different actions to be taken
        # at each time step
        return self._n_actions         

    def observe(self):
        return copy.deepcopy(self._last_observation)  


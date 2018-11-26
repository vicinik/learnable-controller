"""
The environment simulates the behaviour of the system.
The data were measured from the real system.

The inputs of the system are some parameters of the repeaters.
The outputs of the system are the width and the height of the
statistical eye diagram.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys, os, copy, time
from deer.base_classes import Environment

sys.path.insert(1, os.path.join(sys.path[0], '../01_wrappers'))
from data_wrapper import DataWrapper

class MyEnv(Environment):
    def __init__(self, file):
        """ Initialize environment.        
        """
        # Observations = (width, height)
        self._last_observation = [0, 0]
        self._input_dim = [(1,), (1,)]
        self.is_terminal = False
        self._system = DataWrapper(file)
        self._summary_number = 0
        self._best_reward = -9999.0
        self._reset_observation = self._system.get_random_values()

    def act(self, action):
        actions = []
        dimensions = self._system.get_input_dimensions()
        for i in range(len(action)):
            actions.append(int((action[i]+1.0)*((dimensions[i]+1.0)/2)))
        #print('Current actions: {}'.format(actions))
        width, height = self._system.get_eye(actions)
        self._last_observation = [width, height]
        #print('Current observation: {}'.format(self._last_observation))
        reward = - (width + height -4.0)**2
        #print('Current reward: {}'.format(reward))

        # Save best reward until now
        if reward > self._best_reward:
            self._best_reward = reward
            self._reset_observation = self._last_observation

        #time.sleep(1)
        return reward
                
    def reset(self, mode=0):
        self._last_observation = self._reset_observation
        self.is_terminal = False

        return self._last_observation
                
    def inTerminalState(self):
        """Tell whether the environment reached a terminal state after the last transition (i.e. the last transition 
        that occured was terminal).
        """
        return self.is_terminal
        
    def summarizePerformance(self, test_data_set):
        """ This function is called at every PERIOD_BTW_SUMMARY_PERFS.

        Arguments:
            [test_data_set] Simulation data returned by the agent.
        """
        plot_file = 'plots/plot_{}.png'.format(self._summary_number)
        self._summary_number = self._summary_number + 1
        print('Summarize performance: Writing "{}"'.format(plot_file))

        # Get observations
        observations = test_data_set.observations()
        x_observations = observations[0,]
        y_observations = observations[1,]
        # Get all system values
        x_values, y_values = self._system.get_values()

        # Get x achsis of the observations
        observations_x_achsis = []
        for i in range(len(x_observations)):
            for j in range(len(x_values)):
                if abs(x_observations[i] - x_values[j]) < 0.0001 and abs(y_observations[i] - y_values[j]) < 0.0001:
                    observations_x_achsis.append(j)
                    break

        # Plot the values
        fig = plt.figure(figsize=(19,10))
        plt.subplot(211)
        plt.plot(x_values, 'y-')
        plt.plot(observations_x_achsis, x_observations, 'rx')
        plt.subplot(212)
        plt.plot(y_values, 'y-')
        plt.plot(observations_x_achsis, y_observations, 'rx')
        plt.savefig(plot_file)
        plt.close(fig)

        # Reverse the system after a few runs
        if self._summary_number == 10:
            self._system.reverse()

    def inputDimensions(self):
        return self._input_dim

    def nActions(self):
        # The environment allows two different actions to be taken
        # at each time step
        dimensions = self._system.get_input_dimensions()
        actions = []
        for dim in dimensions:
            actions.append((-.99,.99))
        return actions           

    def observe(self):
        return copy.deepcopy(self._last_observation)  


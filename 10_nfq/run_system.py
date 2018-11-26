""" Pendulum environment launcher.
Same principles as run_toy_env. See the docs for more details.

Authors: Vincent Francois-Lavet, David Taralla
"""

import sys
import logging
import numpy as np

import deer.experiment.base_controllers as bc
from deer.default_parser import process_args
from deer.agent import NeuralAgent
from deer.q_networks.AC_net_keras import MyACNetwork
from deer.q_networks.NN_keras_LSTM import NN as NN_keras
from system_env import MyEnv as system_env

class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 10
    EPOCHS = 200
    STEPS_PER_TEST = 100
    PERIOD_BTW_SUMMARY_PERFS = 1

    # ----------------------
    # Environment Parameters
    # ----------------------
    FRAME_SKIP = 1

    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = 0.05
    LEARNING_RATE_DECAY = 0.1
    DISCOUNT = 0.99
    DISCOUNT_INC = .99
    DISCOUNT_MAX = 0.99
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001
    MOMENTUM = 0
    CLIP_DELTA = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = 0
    EPSILON_DECAY = 100
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    FREEZE_INTERVAL = 100
    DETERMINISTIC = True

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Parse parameters ---
    parameters = process_args(sys.argv[1:], Defaults)
    if parameters.deterministic:
        rng = np.random.RandomState(12345)
    else:
        rng = np.random.RandomState()
    
    # --- Instantiate environment ---
    env = system_env('../00_data/4_input_system.csv')

    # --- Instantiate qnetwork ---
    qnetwork = MyACNetwork(
        env,
        parameters.rms_decay,
        parameters.rms_epsilon,
        parameters.momentum,
        parameters.clip_delta,
        parameters.freeze_interval,
        parameters.batch_size,
        parameters.update_rule,
        random_state=rng,
        neural_network_actor=NN_keras,
        neural_network_critic=NN_keras)
    
    # --- Instantiate agent ---
    agent = NeuralAgent(
        env,
        qnetwork,
        parameters.replay_memory_size,
        max(env.inputDimensions()[i][0] for i in range(len(env.inputDimensions()))),
        parameters.batch_size,
        rng)

    # --- Bind controllers to the agent ---
    # For comments, please refer to run_toy_env.py
    agent.attach(bc.VerboseController(
        evaluate_on='epoch', 
        periodicity=1))

    agent.attach(bc.TrainerController(
        evaluate_on='action', 
        periodicity=parameters.update_frequency, 
        show_episode_avg_V_value=True, 
        show_avg_Bellman_residual=True))

    #agent.attach(bc.LearningRateController(
    #    initial_learning_rate=parameters.learning_rate,
    #    learning_rate_decay=parameters.learning_rate_decay,
    #    periodicity=1))

    agent.attach(bc.DiscountFactorController(
        initial_discount_factor=parameters.discount,
        discount_factor_growth=parameters.discount_inc,
        discount_factor_max=parameters.discount_max,
        periodicity=1))

    agent.attach(bc.AutoTuningController(
        exp_r_max=-60,
        exp_r_min=-450,
        l_max=parameters.learning_rate,
        e_max=parameters.epsilon_start,
        e_min=parameters.epsilon_min))

    agent.attach(bc.InterleavedTestEpochController(
        id=0, 
        epoch_length=parameters.steps_per_test, 
        controllers_to_disable=[0, 1, 2, 3], 
        periodicity=10, 
        show_score=True,
        summarize_every=parameters.period_btw_summary_perfs))
    
    # --- Run the experiment ---
    agent.run(parameters.epochs, parameters.steps_per_epoch)

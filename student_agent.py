# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym

try:
    with open("q_table.pkl", "rb") as f:
        Q_table = pickle.load(f)
except FileNotFoundError:
    # Fallback to empty Q-table if file is not found
    Q_table = {}

def get_state(obs):
    """
    Convert the observation into a simplified state representation
    that matches how you trained your Q-table.
    """
    taxi_row, taxi_col = obs[0], obs[1]
    passenger_look = obs[14]  # Example: If passenger is at taxi's location
    return (taxi_row, taxi_col, passenger_look)

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    state = get_state(obs)

    # Fallback to random action if state is not in Q-table
    if state not in Q_table:
        return random.randrange(6)  # 6 possible actions

    # Otherwise choose the action with the highest Q-value
    return np.argmax(Q_table[state])
    # return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.


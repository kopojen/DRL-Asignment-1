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

def get_train_state(obs):
    return obs[10], obs[11], obs[12], obs[13]
    # return (obs[0], obs[1], obs[2], obs[3], obs[4], obs[5], obs[6], obs[7], obs[8], obs[9], obs[10], obs[11], obs[12], obs[13], obs[14], obs[15])

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    state = get_train_state(obs)
    
    # Fallback to random action if state is not in Q-table
    if state not in Q_table:
        return random.randrange(6)  # 6 possible actions

    # Otherwise choose the action with the highest Q-value
    return np.argmax(Q_table[state])
    # return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

import numpy as np
import pickle
import random
from simple_custom_taxi_env import SimpleTaxiEnv

# Hyperparameters
ALPHA = 0.1  # Learning rate
GAMMA = 0.9  # Discount factor
EPISODES = 1  # Training episodes
EPSILON = 0.99  # Exploration rate
q_table = {}

try:
    with open("q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
except FileNotFoundError:
    # Fallback to empty Q-table if file is not found
    q_table = {}
 
count = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  

def rev_softmax(x, temperature=0.5):
    """Compute numerically stable softmax, favoring smaller values (less chosen actions)."""
    x = np.array(x, dtype=np.float64)  # Ensure float64 for precision
    x = x - np.max(x)  # Stability trick to prevent overflow
    exp_x = np.exp(-x / temperature)  # Negative to favor less chosen actions
    
    sum_exp_x = np.sum(exp_x)
    if sum_exp_x == 0:  
        return np.ones_like(x) / len(x)  # If all weights are zero, return uniform probabilities
    
    return exp_x / sum_exp_x  # Normalize

def get_action(state):
    """Choose action using epsilon-greedy policy, favoring less used actions probabilistically."""
    global count

    if state not in q_table:
        q_table[state] = np.zeros(6)

    action_space = [a for a in range(4)]  # Only choose 0 ~ 3
    counts = np.array([count[a] for a in action_space])
    probabilities = rev_softmax(counts)  # Compute softmax probabilities

    action = np.random.choice(action_space, p=probabilities)  # Choose action based on probability
        
    # if random.uniform(0, 1) < EPSILON:  # Exploration
    #     counts = np.array([count[a] for a in action_space])
    #     probabilities = rev_softmax(counts)  # Compute softmax probabilities

    #     action = np.random.choice(action_space, p=probabilities)  # Choose action based on probability
    # else:  # Exploitation
    #     action = np.argmax(q_table[state])

    count[action] += 1  
    return action
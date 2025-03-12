import pickle
import random
import numpy as np
from simple_custom_taxi_env import SimpleTaxiEnv  # Make sure the filename/path matches

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95
EPSILON = 1.0
MIN_EPSILON = 0.05
DECAY_RATE = 0.999
EPISODES = 100

env = SimpleTaxiEnv(grid_size=5, fuel_limit=5000)
action_size = 6  # [Down, Up, Right, Left, Pickup, Dropoff]
Q_table = {}

def get_state(obs):
    """
    Convert observation into a simplified state representation 
    that is hashable (tuple of ints).
    """
    # Example: taxi_row, taxi_col, passenger_look
    taxi_row, taxi_col = obs[0], obs[1]
    passenger_look = obs[14]  # obs[14] indicates whether passenger is at your location or not
    return (taxi_row, taxi_col, passenger_look)

for episode in range(EPISODES):
    obs, _ = env.reset()
    state = get_state(obs)
    done = False
    total_reward = 0

    while not done:
        # Epsilon-greedy action selection
        if random.random() < EPSILON:
            action = random.randrange(action_size)
        else:
            if state in Q_table:
                action = np.argmax(Q_table[state])
            else:
                # Unknown state -> random action
                action = random.randrange(action_size)

        next_obs, reward, done, _ = env.step(action)
        next_state = get_state(next_obs)

        # Update Q-table
        if state not in Q_table:
            Q_table[state] = np.zeros(action_size)
        if next_state not in Q_table:
            Q_table[next_state] = np.zeros(action_size)

        Q_table[state][action] = (1 - LEARNING_RATE) * Q_table[state][action] + \
                                 LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(Q_table[next_state]))

        state = next_state
        total_reward += reward
    
    # Decay epsilon after each episode
    EPSILON = max(MIN_EPSILON, EPSILON * DECAY_RATE)

    if episode % 200 == 0 and episode != 0:
        print(f"Episode {episode}, Total Reward: {total_reward}")

# Save Q-table
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q_table, f)

print("Q-learning training complete. Q-table saved to 'q_table.pkl'.")
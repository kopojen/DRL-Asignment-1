import pickle
import random
import numpy as np
from simple_custom_taxi_env import SimpleTaxiEnv  # Make sure the filename/path matches

# Hyperparameters
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.99
EPSILON = 1.0
MIN_EPSILON = 0.05
DECAY_RATE = 0.9999
EPISODES = 100000

env = SimpleTaxiEnv(grid_size=5, fuel_limit=50)
action_size = 6  # [Down, Up, Right, Left, Pickup, Dropoff]
Q_table = {}

# def get_state(obs):
#     """
#     Convert observation into a simplified state representation 
#     that is hashable (tuple of ints).
#     """
#     # Example: taxi_row, taxi_col, passenger_look
#     taxi_row, taxi_col = obs[0], obs[1]
#     passenger_look = obs[14]  # obs[14] indicates whether passenger is at your location or not
#     return (taxi_row, taxi_col, passenger_look)

def get_state(obs):
    """Return an enhanced environment state representation."""
    taxi_row, taxi_col = obs[0], obs[1]

    # Obstacles (if included in obs)
    obstacle_north, obstacle_south, obstacle_east, obstacle_west = obs[10:14]
    
    passenger_look = obs[14]
    destination_look = obs[15]

    # Encoded state tuple
    state = (
        taxi_row, taxi_col,
        obstacle_north, obstacle_south, obstacle_east, obstacle_west, 
        passenger_look, destination_look
    )
    return state

def reward_shaping(env, old_obs, new_obs):
    shaped_reward = 0
    
    old_pick_dis = abs(old_obs[0] - env.passenger_loc[0]) + abs(old_obs[1] - env.passenger_loc[1])
    new_pick_dis = abs(new_obs[0] - env.passenger_loc[0]) + abs(new_obs[1] - env.passenger_loc[1])
    
    if not env.passenger_picked_up and old_pick_dis <= new_pick_dis:    # not going toward the passenger
        shaped_reward -= 2
        
    old_drop_dis = abs(old_obs[0] - env.destination[0]) + abs(old_obs[1] - env.destination[1])
    new_drop_dis = abs(new_obs[0] - env.destination[0]) + abs(new_obs[1] - env.destination[1])
    
    if env.passenger_picked_up and old_pick_dis <= new_pick_dis:    # not going toward the passenger
        shaped_reward -= 2
        
    return shaped_reward

rewards_per_episode = []
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
                
        taxi_row, taxi_col = env.taxi_pos
        next_row, next_col = taxi_row, taxi_col
        reward = 0
        if action == 0 :  # Move Down
            next_row += 1
        elif action == 1:  # Move Up
            next_row -= 1
        elif action == 2:  # Move Right
            next_col += 1
        elif action == 3:  # Move Left
            next_col -= 1
        
        while (taxi_row + next_row, taxi_col + next_col) in env.obstacles:
            action = random.randrange(action_size)
            
        next_obs, reward, done, _ = env.step(action)
        next_state = get_state(next_obs)
        
        shaped_reward = reward_shaping(env, obs, next_obs)
        
        # Update Q-table
        if state not in Q_table:
            Q_table[state] = np.zeros(action_size)
        if next_state not in Q_table:
            Q_table[next_state] = np.zeros(action_size)

        Q_table[state][action] = (1 - LEARNING_RATE) * Q_table[state][action] + \
                                 LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(Q_table[next_state]))

        state = next_state
        
        reward += shaped_reward
        total_reward += reward
        
    rewards_per_episode.append(total_reward)
    # Decay epsilon after each episode
    EPSILON = max(MIN_EPSILON, EPSILON * DECAY_RATE)

    if (episode + 1) % 200 == 0:
        avg_reward = np.mean(rewards_per_episode[-100:])
        print(f"🚀 Episode {episode + 1}/{EPISODES}, Average Reward: {avg_reward:.2f}, Epsilon: {EPSILON:.3f}")

# Save Q-table
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q_table, f)

print("Q-learning training complete. Q-table saved to 'q_table.pkl'.")
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


def get_direction(taxi_r, taxi_c, station_r, station_c):
    x = station_r - taxi_r
    y = station_c - taxi_c
    if x == 0 and y == 0:
        return 8 #站上
    elif x > 0 and y > 0:
        return 0 # 第一象限
    elif x > 0 and y < 0:
        return 1 #第二象限
    elif x < 0 and y > 0:
        return 2 #第三象限
    elif x < 0 and y < 0:
        return 3 # 第四象限
    elif x == 0 and y > 0:
        return 4 #東
    elif x == 0 and y < 0:
        return 5 #西
    elif x > 0 and y == 0:
        return 6 #南
    elif x < 0 and y == 0:
        return 7 #北
    
def get_train_state(obs, passenger_on_taxi, target_station):
    (taxi_r, taxi_c,
    sO_r, sO_c, s1_r, s1_c, s2_r, s2_c, s3_r, s3_c,
    obs_n, obs_s, obs_e, obs_w,
    passenger_look, destination_look) = obs
    
    station_positions = [(sO_r, sO_c), (s1_r, s1_c), (s2_r, s2_c), (s3_r, s3_c)]

    station_r, station_c = station_positions [target_station]
    direction = get_direction (taxi_r, taxi_c, station_r, station_c)
    
    # can pickup: no passenger on taxi and passenger_look == 1
    new_pickup = int( (not passenger_on_taxi) and (passenger_look == 1) and ((taxi_r, taxi_c) in station_positions))
    # can dropoff: passenger on taxi and destination_look == 1
    new_dropoff = int (passenger_on_taxi and (destination_look == 1) and ((taxi_r, taxi_c) in station_positions))
    
    return (
    direction,
    int (obs_n), int(obs_s), int (obs_e), int(obs_w),
    new_pickup, new_dropoff)

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

def is_near_station(taxi_pos, station_pos):
    taxi_x, taxi_y = taxi_pos
    station_x, station_y = station_pos
    return abs(taxi_x - station_x) + abs(taxi_y - station_y) == 1  

def get_action(obs):
    
    # TODO: Train your own agent
    # HINT: If you're using a Q-table, consider designing a custom key based on `obs` to store useful information.
    # NOTE: Keep in mind that your Q-table may not cover all possible states in the testing environment.
    #       To prevent crashes, implement a fallback strategy for missing keys. 
    #       Otherwise, even if your agent performs well in training, it may fail during testing.
    
    global passenger_on_taxi, current_station_ix
    passenger_on_taxi, current_station_ix = 0, 0
    (taxi_r, taxi_c,
    s0_r, s0_c, s1_r, s1_c, s2_r, s2_c, s3_r, s3_c,
    obs_n, obs_s, obs_e, obs_w,
    p_look, d_look) = obs

    station_positions = [
        (s0_r, s0_c),
        (s1_r, s1_c),
        (s2_r, s2_c),
        (s3_r, s3_c)
    ]
    target_station = current_station_ix
    state = get_train_state(obs, passenger_on_taxi, target_station)
    taxi_pos = (taxi_r, taxi_c)
    
    action_space = [a for a in range(4)]  # Only choose 0 ~ 3
    counts = np.array([count[a] for a in action_space])
    probabilities = rev_softmax(counts)  # Compute softmax probabilities

    # action = np.random.choice(action_space, p=probabilities)  # Choose action based on probability
        
    if state not in Q_table:
        counts = np.array([count[a] for a in action_space])
        probabilities = rev_softmax(counts)  # Compute softmax probabilities

        action = np.random.choice(action_space, p=probabilities)  # Choose action based on probability
    else:  # Exploitation
        print("innnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnnn")
        action = np.argmax(Q_table[state])
        
    if action == 4:  # pickup
        if not passenger_on_taxi and p_look == 1 and taxi_pos in station_positions:
            passenger_on_taxi = 1
            current_station_ix = 0 
    elif action == 5:  # dropoff
        if passenger_on_taxi and d_look == 1 and taxi_pos in station_positions:
            passenger_on_taxi = 0

    if ((not passenger_on_taxi and p_look != 1) or (passenger_on_taxi and d_look != 1)) \
    and is_near_station(taxi_pos, station_positions[current_station_ix]):
        current_station_ix = (current_station_ix + 1) % 4
    
    return action
    # return random.choice([0, 1, 2, 3, 4, 5]) # Choose a random action
    # You can submit this random agent to evaluate the performance of a purely random strategy.

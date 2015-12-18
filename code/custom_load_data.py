import csv
import numpy as np

def load_

def load_trajectories(feat_path, target_action):
    with open(file_name) as samples:
        file_reader = csv.DictReader(
                samples,
                fieldnames = ['user', 'action', 'reward'] ,
                restkey = 'state',
                delimiter = ',')
        state_values = next(file_reader, None)['state']
        trajectories = []
        current_trajectory = []
        last_states = [0, 0, 0, frozenset([])]
        previous_user = None
        for row in file_reader:
            # grab only the states which are 1s
            state = tuple(map(lambda x: int(x), row['state']))
            user = row['user']
            action = row['action']
            reward = row['reward']
            if user != previous_user and current_trajectory:
                trajectories.append(current_trajectory)
                current_trajectory = []
                last_states =
            previous_user = user
            last_states = [last_states[3], action, reward, state]
            current_trajectory.append(last_states)
        if current_trajectory:
            trajectories.append(current_trajectory)
        return trajectories
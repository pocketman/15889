import csv
import numpy as np
from sample_policy import SamplePolicy

def get_sample_policy(file_name):
    action_counts = get_action_counts(file_name)
    policy = convert_action_counts_to_policy(action_counts)
    return policy

def load_trajectories(file_name):
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
            state = map(
                    lambda x, y: y if x == '1' else None,
                    row['state'],
                    state_values)
            state = filter(lambda x: x != None, state)

            state = frozenset(state)
            user = row['user']
            action = row['action']
            reward = row['reward']
            if user != previous_user and current_trajectory:
                trajectories.append(current_trajectory)
                current_trajectory = []
                last_states == [0, 0, 0, frozenset([])]
            previous_user = user
            last_states = [last_states[3], action, reward, state]
            current_trajectory.append(last_states)
        if current_trajectory:
            trajectories.append(current_trajectory)
        return trajectories

def get_action_counts(file_name):
    action_counts = dict([]);
    with open(file_name) as sample_file:
        file_reader = csv.DictReader(
                sample_file,
                fieldnames = ['user', 'action', 'reward'] ,
                restkey = 'state',
                delimiter = ',')
        state_values = next(file_reader, None)['state']
        prev_state = frozenset([])
        for row in file_reader:
            # grab only the states which are 1s
            next_state = map(
                    lambda x, y: y if x == '1' else None,
                    row['state'],
                    state_values)
            next_state = filter(lambda x: x != None, next_state)

            next_state = frozenset(next_state)
            action = row['action']
            if prev_state in action_counts:
                if action in action_counts[prev_state]:
                    action_counts[prev_state][action] = action_counts[prev_state][action] + 1
                else:
                    action_counts[prev_state][action] = 1
            else:
                action_counts[prev_state] = {action: 1}
            prev_state = next_state
    return action_counts

def convert_action_counts_to_policy(action_counts):
    return SamplePolicy(action_counts)

def estimate_utility(sample_policy, test_policy, trajectories, discount):
    """
    runs importance sampling to calculate expected value of test_policy using
    importance sampling on the sample_policy
    """
    utility = 0;
    for traj in trajectories:
        t = 0
        p = 1
        while t < len(traj) and p > 0:
            s_t = traj[t][0]
            a_t = traj[t][1]
            r_t = float(traj[t][2])
            p_test = test_policy.p_action_given_state(a_t, s_t)
            p_sample = sample_policy.p_action_given_state(a_t, s_t)
            p = p_test / p_sample * p
            utility += discount ** t * r_t * p
            t = t + 1
    return utility

def estimate_utility_for_single_trajectory(
        sample_policy,
        test_policy,
        trajectory,
        discount):
    raise Exception('Doesn\'t work')
    utility = 0
    t = 0
    p = 1
    while t < len(trajectory) and p > 0:
        s_t = trajectory[t][0]
        a_t = trajectory[t][1]
        r_t = trajectory[t][2]
        p_test = test_policy.p_ction_given_state(a_t, s_t, test_policy)
        p_sample = sample_policy.p_action_given_state(a_t, s_t, sample_policy)
        p = p_test / p_sample * p
        utility += discount ** t * r_t * p
        t = t + 1
    return utility

def hcope(X, c, delta):
    """
    X - list of X_i >= 0 such that E[X_i] == u
    c - the cut off we want to use for hcope
    delta - probability bound we want on our output
    returns the reward such that E[X_i] >= output with probability 1 - delta
    """
    n = len(X)
    Y = np.empty(n)
    Y.fill(c)
    Y = np.minimum(X, Y)
    lbound = (np.mean(Y) - 7 * c * np.log(2 / delta) / (3 * (n - 1)) -
            np.sqrt(2 * np.log(2 / delta) * np.variance(Y) / n))
    return lbound
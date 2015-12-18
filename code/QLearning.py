import numpy as np
import filter_data as fd
EPS = 0.001
labels_path = 'C:\\Users\\REX\\Dropbox\\cmu\\fall2015\\15889\\project\\results\\d0.9-ur0.7-e10-i1000-tQ315\\valid_actions.txt'

labels = np.array(fd.get_labels(labels_path))

def estimate_rewards(next_states, actions, rewards, reward_action, v = 1):
    """
    next_states - array of binary vectors representing our states after executing action a
    actions - array of actions (as indices) representing the action we took
    rewards - represents the reward we got after transitioning into this states
    reward_action - action which transitions us to a reward state
    returns the mean reward for each reward state
    """
    avg_reward = estimate_empirical_reward(actions, rewards, reward_action)
    next_states = next_states[actions == reward_action]
    rewards = rewards[actions == reward_action]
    actions = actions[actions == reward_action]
    state_total_rewards ={}
    for i in range(len(next_states)):
        s = tuple(next_states[i])
        r = rewards[i]
        if s in state_total_rewards:
            state_total_rewards[s].append(r)
        else:
            state_total_rewards[s] = [r]
    for s in state_total_rewards.keys():
        r = np.array(state_total_rewards[s])
        #if np.mean(r) > 3 and len(r) > 5:
        #    print masked_feat_labels[np.array(s).astype(int) == 1], str(r)
        state_total_rewards[s],_ = update_prior_mean_with_data(0, v, r)
    return state_total_rewards

def discount_rewards(state_rewards, discount):
    """
    state_rewards - dictionary mapping state to reward for that state
    returns the mapping of state to reward to be discounted by the shortest path to
    that reward state
    """
    discounted_rewards = {}
    for s in state_rewards.keys():
        path_length = np.sum(np.array(s))
        discounted_rewards[s] = discount ** path_length * state_rewards[s]
    return discounted_rewards

def get_max_reward_states(discounted_state_rewards):
    """
    gets the states with max discounted reward
    """
    max_states = []
    max_reward = -1
    for s in discounted_state_rewards.keys():
        if discounted_state_rewards[s] > max_reward + EPS:
            max_reward = discounted_state_rewards[s]
            max_states = [s]
        elif np.abs(discounted_state_rewards[s] - max_reward) < EPS:
            max_states.append(s)
    return max_states

def estimate_empirical_reward(actions, rewards, reward_action):
    action_rewards = rewards[actions == reward_action]
    return np.sum(action_rewards) / len(action_rewards)

def update_prior_mean_with_data(u_0, v_0, X):
    """
    Assumes a normal gamma distribution for our mean.
    u_0 - mean of prior
    v_0 - number of observations used to estimate u_0
    X - array of data points which we observed
    Returns the updated mean and updated v
    """
    return float(v_0 * u_0 + np.sum(X)) / (v_0 + len(X)), v_0 + len(X)
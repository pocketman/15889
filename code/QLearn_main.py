'''
@author: Ruixin Li
'''
from fqi_policy_yohan import FQIPolicy
from sample_policy_yohan import SamplePolicy
from sklearn.externals import joblib
from load_data import load_data
from collections import defaultdict
from importance_sampler import *
import QLearning as ql
import filter_data as fd

feat_path = "C:\\Users\\REX\\Dropbox\\cmu\\fall2015\\15889\\project\\lectures\\feats.csv"
labels_path = 'C:\\Users\\REX\\Dropbox\\cmu\\fall2015\\15889\\project\\results\\d0.9-ur0.7-e10-i1000-tQ315\\valid_actions.txt'
target_action = "Q315"
discount = 0.99
labels = np.array(fd.get_labels(labels_path))

cur_states, actions, rewards, next_states, users, action_index, user_index, valid_feats = load_data(
    feat_path,
    target_action,
    num_users_ratio = 1,
    valid_feats = [1] * len(labels))
print cur_states.shape, actions.shape, rewards.shape, next_states.shape, len(user_index)

print "!1"
action_counts = defaultdict(lambda: defaultdict(int))
for s,a in zip(cur_states,actions):
    action_counts[tuple(s)][a] += 1

print "!2"
trajectories = []
prev_u = None
for s,a,r,u in zip(cur_states,actions,rewards,users):
    if u != prev_u: trajectories.append([])
    trajectories[-1].append((s,a,r))
"""
print "!3"
sample_policy = SamplePolicy(action_counts)
test_policy = FQIPolicy(approximator, top_k)
u = estimate_utility(sample_policy, test_policy, trajectories, discount)

print u
delta = 0.05
lower_bound = hcope(sample_policy, test_policy, trajectories, discount, delta)
print 'Lower bound (p=0.95) of {lb}'.format(lb = lower_bound)
"""

state_rewards = ql.estimate_rewards(next_states, actions, rewards, action_index[target_action])
discounted_rewards = ql.discount_rewards(state_rewards, discount)
discounted_max_states = ql.get_max_reward_states(discounted_rewards)
max_states = ql.get_max_reward_states(state_rewards)

print 'Reward of max state = {a}, discounted max state = {b}'.format(
    a = state_rewards[max_states[0]], b = state_rewards[discounted_max_states[0]])
print 'Discounted reward of max state = {a}, discounted max state = {b}'.format(
    a = discounted_rewards[max_states[0]], b = discounted_rewards[discounted_max_states[0]])

print 'Max state actions: {a} \nDiscounted max state actions: {b}'.format(
    a = labels[np.array(max_states[0]).astype(int) == 1], b = labels[np.array(discounted_max_states[0]).astype(int) == 1])
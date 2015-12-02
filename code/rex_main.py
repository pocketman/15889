'''
Created on Nov 25, 2015

@author: Yohan
'''
from fqi_policy_yohan import FQIPolicy
from sample_policy_yohan import SamplePolicy
from sklearn.externals import joblib
from load_data import load_data
from collections import defaultdict
from importance_sampler import *

feat_path = "C:\\Users\\REX\\Dropbox\\cmu\\fall2015\\15889\\project\\lectures\\feats.csv"
result_dir = "C:\\Users\\REX\\Dropbox\\cmu\\fall2015\\15889\\project\\results\\d0.9-ur0.7-e10-i1000-tQ315"
approximator_path = result_dir+"\\approximator\\random_forest_regressor.model"
target_action = "Q315"
discount = 0.9
num_users = 9999999  # max num of users to consider
top_k = 10  # optimal policy (probability over top k actions. top_k=-1 means all actions)

with open(result_dir+"/valid_actions.txt") as in_file: action_list = in_file.read().strip().split("\n")
with open(result_dir+"/valid_users.txt") as in_file: exclude_users = set(in_file.read().strip().split("\n"))
with open(result_dir+"/valid_feats.txt") as in_file: valid_feats = map(int,in_file.read().strip().split("\n"))

approximator = joblib.load(approximator_path)
cur_states, actions, rewards, next_states, users, action_index, user_index, valid_feats = load_data(feat_path, target_action, num_users=num_users, exclude_users=exclude_users, valid_feats=valid_feats, valid_actions=action_list)
# cur_states, actions, rewards, next_states, users, action_index, user_index, valid_feats = load_data(feat_path, target_action, num_users=num_users, valid_feats=valid_feats, valid_actions=action_list)
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

print "!3"
sample_policy = SamplePolicy(action_counts)
test_policy = FQIPolicy(approximator, top_k)
u = estimate_utility(sample_policy, test_policy, trajectories, discount)
# u = estimate_utility(sample_policy, sample_policy, trajectories, discount)
print u
delta = 0.05
lower_bound = hcope(sample_policy, test_policy, trajectories, discount, delta)
print 'Lower bound (p=0.95) of {lb}'.format(lb = lower_bound)
'''
Created on Nov 23, 2015

@author: Yohan
'''
from random_forest import RandomForest
import numpy as np
from load_data import *
import sys
import csv
import os

discount = 0.9
num_estimators = 10  # estimators inside the random forest 
num_iters = 1000  # num of training iterations
num_users = 12000  # max num of users(=trajectories) (<= sys.maxint)
target_action = "Q235"  # every trajectory should end with this action
feat_path = "../../data/lectures/feats.csv"
result_dir = "../results/d"+str(discount)+"-u"+str(num_users)+"-e"+str(num_estimators)+"-i"+str(num_iters)+"-t"+target_action
approximator_path = result_dir + "/approximator/random_forest_regressor.model"  # path to save the trained approximator
debug_action_cnt = False
debug_q0 = True

# Load data
cur_states, actions, rewards, next_states, action_index, valid_users, valid_feats = load_data(feat_path, target_action, num_users=num_users)
print cur_states.shape, next_states.shape, actions.shape, rewards.shape, len(action_index), len(valid_users), len(valid_feats)
# dim: cur_states,next_states = num_instances x num_features 
# dim: actions,rewards = num_instances

num_feats = cur_states.shape[1]
num_actions = len(action_index)
action_list = [ 0 for x in range(num_actions) ]
for a,i in action_index.iteritems(): action_list[i] = a


s0 = np.zeros((1,num_feats))
approximator = RandomForest(num_estimators=num_estimators, num_actions=num_actions)
approximator.train(cur_states, actions, rewards)
for iter in range(num_iters):
    print "Iteration", iter
    qs = approximator.predict(next_states)  # dim: num_actions x num_instances
    max_qs = np.amax(qs, axis=0)  # dim: num_instances
    approximator.train(cur_states, actions, rewards + discount * max_qs)
    
    if debug_action_cnt:
        max_as = np.argmax(qs, axis=0)
        action_cnt = defaultdict(int)
        for a in max_as: action_cnt[action_list[a]] += 1
        print "action count: ", action_cnt
    if debug_q0:
        q0 = approximator.predict(s0)
        print "max q0:", np.amax(q0, axis=0)[0], ", max action:", action_list[np.argmax(q0, axis=0)]

# print
if not os.path.exists(result_dir): 
    os.mkdir(result_dir)
    os.mkdir(result_dir+"/approximator")
approximator.save(approximator_path)
with open(result_dir+"/valid_actions.txt","w") as out_file: print >> out_file, "\n".join(action_list)
with open(result_dir+"/valid_users.txt","w") as out_file: print >> out_file, "\n".join(valid_users)
with open(result_dir+"/valid_feats.txt","w") as out_file: print >> out_file, "\n".join(map(str,np.array(valid_feats,dtype=int)))
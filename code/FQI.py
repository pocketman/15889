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
import time

discount = 0.9
num_estimators = 10  # estimators inside the random forest 
num_iters = 1000  # num of training iterations
num_users = sys.maxint  # max num of users(=trajectories) (use sys.maxint for unlimited case)
num_users_ratio = 0.7  # % of users(=trajectories) for training
demography = False
target_action = "Q235"  # every trajectory should end with this action
# target_action = "Q315"  # every trajectory should end with this action
feat_path = "../../data/lectures+demography/feats.csv"
result_dir = "../results/demo"+("1" if demography else '0')+"-d"+str(discount)+("-un"+str(num_users) if num_users != sys.maxint else "")+("-ur"+str(num_users_ratio) if num_users_ratio != 1 else "")+"-e"+str(num_estimators)+"-i"+str(num_iters)+"-t"+target_action
approximator_path = result_dir + "/approximator/random_forest_regressor.model"  # path to save the trained approximator
debug_action_cnt = False
debug_q0 = True


# Load data
cur_states, actions, rewards, next_states, users, action_index, user_index, valid_feats = load_data(feat_path, target_action, num_users=num_users, num_users_ratio=num_users_ratio, demography=demography)
print cur_states.shape, next_states.shape, actions.shape, rewards.shape, users.shape, len(action_index), len(user_index), sum(valid_feats)
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
    print "---------------------------------------------------------------\nIteration", iter
    start_time = time.time()
    
    qs = approximator.predict(next_states)  # dim: num_actions x num_instances
    max_qs = np.amax(qs, axis=0)  # dim: num_instances
    max_qs[actions==action_index[target_action]] = 0
    approximator.train(cur_states, actions, rewards + discount * max_qs)
    
    if debug_action_cnt:
        max_as = np.argmax(qs, axis=0)
        action_cnt = defaultdict(int)
        for a in max_as: action_cnt[action_list[a]] += 1
        print "action count: ", action_cnt
    if debug_q0:
        q0 = approximator.predict(s0)
        print "max q0:", np.amax(q0, axis=0)[0], ", max action:", action_list[np.argmax(q0, axis=0)]
 
    print "Estimated time: ", int((time.time()-start_time)/60*(num_iters-iter-1)), "mins"
 
 
# print
if not os.path.exists(result_dir): 
    os.mkdir(result_dir)
    os.mkdir(result_dir+"/approximator")
approximator.save(approximator_path)
with open(result_dir+"/valid_actions.txt","w") as out_file: print >> out_file, "\n".join(action_list)
with open(result_dir+"/valid_users.txt","w") as out_file: print >> out_file, "\n".join([ u for u,i in sorted(user_index.iteritems(), key=lambda (u,i): i) ])
with open(result_dir+"/valid_feats.txt","w") as out_file: print >> out_file, "\n".join(map(str,np.array(valid_feats,dtype=int)))
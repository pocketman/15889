'''
Created on Nov 23, 2015

@author: Yohan
'''
import csv
from collections import defaultdict
import numpy as np
import sys

def iter_csv(path, header=None):
    in_file = open(path)
    in_csv = csv.reader(in_file)
    local_header = in_csv.next()
    if header:  
        for h in local_header: header.append(h)  # copy header
    for row in in_csv:
        yield row
    in_file.close()
    
    
def load_data(feat_path, target_action, num_users=sys.maxint, exclude_users=set(), valid_feats=None, valid_actions=None):
    # valid_feats = [ 1,0,0,1,... ]
    # valid_actions = [ "L1","L3",... ]
    
    print "Loading data... (usually takes ~30 secs)"
    num_org_feats = len(csv.reader(open(feat_path)).next()) - 3  # exclude "user","action","reward"
    
    valid_users = set()  # users who are included
    user_traj_len = defaultdict(int)
    user_traj = defaultdict(list)  # temporary
    header = []
    for row in iter_csv(feat_path, header):
        if row[0] in exclude_users: continue
        if row[0] in valid_users: continue  # continue if target_action has already been reached
        if valid_actions and row[1] not in valid_actions: continue  # continue if action is not in the given valid_actions
        
        feat = np.array(row[3:], dtype=float)
        user_traj[row[0]].append( (row[1],float(row[2]),feat) )
        user_traj_len[row[0]] += 1
        if row[1] == target_action:
            valid_users.add(row[0])
        
        if len(valid_users) == num_users: break


    # valid actions
    action_index = dict()
    if valid_actions:
        for i, a in enumerate(valid_actions):
            action_index[a] = i
    else:
        tmp_actions = []
        valid_actions = set()
        for user,traj in user_traj.iteritems():
            if user not in valid_users: continue
            for a,r,s in traj:
                tmp_actions.append(a)  # string
                valid_actions.add(a)
        for i, a in enumerate(sorted(valid_actions)):
            action_index[a] = i

    # valid feats
    if valid_feats:
        valid_feats = np.array(valid_feats, dtype=np.bool_)
    else:
        feat_sum = np.zeros(num_org_feats)  # num of occurrences for each feature
        for user,traj in user_traj.iteritems():
            if user not in valid_users: continue
            for a,r,s in traj:
                feat_sum += s
        valid_feats = feat_sum > 0
    num_valid_feats = sum(valid_feats)
   
    num_instances = sum([ length for user,length in user_traj_len.iteritems() if user in valid_users ])
    
    s0 = np.zeros(num_valid_feats)
    cur_states = np.empty((num_instances,num_valid_feats))
    next_states = np.empty((num_instances,num_valid_feats))
    rewards = np.empty(num_instances)
    actions = np.empty(num_instances)
    i = 0
    for user,traj in user_traj.iteritems():
        if user not in valid_users: continue
        for t in range(len(traj)):
            if t==0: cur_states[i,:] = s0
            else: cur_states[i,:] = traj[t-1][2][valid_feats]
            actions[i] = action_index[traj[t][0]]
            if t==len(traj)-1: rewards[i] = traj[t][1]
            else: rewards[i] = 0
            next_states[i,:] = traj[t][2][valid_feats]
            i += 1
    
    return cur_states, actions, rewards, next_states, action_index, valid_users, valid_feats
            
    
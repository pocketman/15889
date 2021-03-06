'''
Created on Nov 23, 2015

@author: Yohan
'''
import csv
from collections import defaultdict
import numpy as np
import sys
import time

def iter_csv(path, header=None):
    in_file = open(path)
    in_csv = csv.reader(in_file)
    local_header = in_csv.next()
    if header is not None:
        for h in local_header: header.append(h)  # copy header
    for row in in_csv:
        yield row
    in_file.close()


# with open(result_dir+"/valid_actions.txt") as in_file: action_list = in_file.read().strip().split("\n")
# with open(result_dir+"/valid_users.txt") as in_file: valid_users = set(in_file.read().strip().split("\n"))
# with open(result_dir+"/valid_feats.txt") as in_file: valid_feats = map(int,in_file.read().strip().split("\n"))


def read_feats(out_dir, mode, target_action):   # mode: {"train","dev","test"}
    cur_state_list = []
    next_state_list = []
    action_list = []
    reward_list = []
    trajectories = []

    cur_trajectory = []
    for row in iter_csv(out_dir+"/feats_"+mode+".csv"):
        cur_state = tuple(map(lambda x: float(x), row[:((len(row)-2)/2)]))
        action = int(row[((len(row)-2)/2)])
        reward = float(row[((len(row)-2)/2)+1])
        next_state = tuple(map(lambda x: float(x), row[((len(row)-2)/2+2):]))
        cur_state_list.append(cur_state)
        action_list.append(action)
        reward_list.append(reward)
        next_state_list.append(next_state)
        cur_trajectory.append((cur_state, action, reward, next_state))
        if action == target_action:
            trajectories.append(cur_trajectory)
            cur_trajectory = []
    cur_states = np.array(cur_state_list)
    actions = np.array(action_list)
    rewards = np.array(reward_list)
    next_states = np.array(next_state_list)

    valid_actions = open(out_dir+"/actions.txt").read().strip().split("\n")
    valid_users = open(out_dir+"/users_"+mode+".txt").read().strip().split("\n")
    return np.array(cur_states), np.array(actions), np.array(rewards), np.array(next_states), np.array(valid_actions), np.array(valid_users), trajectories

def load_data(feat_path, target_action, num_users=sys.maxint, num_users_ratio=1, exclude_users=set(), valid_feats=None, valid_actions=None, demography=False):
    # valid_feats = [ 1,0,0,1,... ]
    # valid_actions = [ "L1","L3",... ]
    assert num_users==sys.maxint or num_users_ratio==1
    demo_start = 3
    feat_start = 3

    print "Loading data... (usually takes ~30 secs)"
    if demography: num_org_feats = len(csv.reader(open(feat_path)).next()) - demo_start
    else: num_org_feats = len(csv.reader(open(feat_path)).next()) - feat_start


    valid_users = set()  # users who are included
    valid_user_index = dict()
    user_traj_len = defaultdict(int)
    user_traj = defaultdict(list)  # temporary
    header = []
    for row in iter_csv(feat_path, header):
        if row[0] in exclude_users: continue
        if row[0] in valid_users: continue  # continue if target_action has already been reached
        if valid_actions and row[1] not in valid_actions: continue  # continue if action is not in the given valid_actions

        if demography: feat = np.array(row[demo_start:], dtype=float)
        else: feat = np.array(row[feat_start:], dtype=float)
        user_traj[row[0]].append( (row[1],float(row[2]),feat) )
        user_traj_len[row[0]] += 1
        if row[1] == target_action:
            valid_users.add(row[0])

        if len(valid_users) == num_users: break

    np.random.seed(int(time.time()))
    valid_users = set(filter(lambda u: np.random.rand() < num_users_ratio, valid_users))
    for u in valid_users:
        valid_user_index[u] = len(valid_user_index)


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
    actions = np.empty(num_instances, dtype=int)
    users = np.empty(num_instances)
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
            users[i] = valid_user_index[user]
            i += 1

    return cur_states, actions, rewards, next_states, users, action_index, valid_user_index, valid_feats
    # dim: cur_states,next_states = num_instances x num_features
    # dim: actions,rewards = num_instances


def load_data_int(feat_path, target_action, num_users=sys.maxint, num_users_ratio=1, exclude_users=set(), valid_feats=None, valid_actions=None, demography=False):
    # valid_feats = [ 1,0,0,1,... ]
    # valid_actions = [ "L1","L3",... ]
    assert num_users==sys.maxint or num_users_ratio==1
    demo_start = 3
    feat_start = 3

    print "Loading data... (usually takes ~30 secs)"
    if demography: num_org_feats = len(csv.reader(open(feat_path)).next()) - demo_start
    else: num_org_feats = len(csv.reader(open(feat_path)).next()) - feat_start


    valid_users = set()  # users who are included
    user_traj_len = defaultdict(int)
    user_traj = defaultdict(list)  # temporary
    header = []
    for row in iter_csv(feat_path, header):
        if row[0] in exclude_users: continue
        if row[0] in valid_users: continue  # continue if target_action has already been reached
        if valid_actions and row[1] not in valid_actions: continue  # continue if action is not in the given valid_actions

        if demography: feat = np.array(row[demo_start:], dtype=float)
        else: feat = np.array(row[feat_start:], dtype=float)
        user_traj[row[0]].append( (row[1],float(row[2]),feat) )
        user_traj_len[row[0]] += 1
        if row[1] == target_action:
            valid_users.add(row[0])

        if len(valid_users) == num_users: break

    action_index_orig = {a: i for i, a in enumerate(header[feat_start:])}
    np.random.seed(int(time.time()))
    valid_users = set(filter(lambda u: np.random.rand() < num_users_ratio, valid_users))
    valid_user_index = {u: i for i, u in enumerate(valid_users)}


    # valid actions
    if valid_actions is None:
        tmp_actions = []
        valid_actions = set()
        for user,traj in user_traj.iteritems():
            if user not in valid_users: continue
            for a,r,s in traj:
                tmp_actions.append(a)  # string
                valid_actions.add(a)
    action_index = {a: i for i, a in enumerate(sorted(valid_actions))}

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

    valid_action_index = []
    sum_ = 0
    for vf in valid_feats:
        valid_action_index.append(sum_)
        sum_ += vf
    cur_states = np.empty((num_instances,num_valid_feats))
    next_states = np.empty((num_instances,num_valid_feats))
    rewards = np.empty(num_instances)
    actions = np.empty(num_instances, dtype=int)
    users = np.empty(num_instances)
    i = 0
    for user,traj in user_traj.iteritems():
        if user not in valid_users: continue
        s = np.zeros(num_valid_feats)
        for t in range(len(traj)):
            cur_states[i,:] = s
            actions[i] = action_index[traj[t][0]]
            vai = action_index_orig[traj[t][0]]
            s[valid_action_index[vai]] += 1
            if t==len(traj)-1: rewards[i] = traj[t][1]
            else: rewards[i] = 0
            next_states[i,:] = s
            users[i] = valid_user_index[user]
            i += 1

    return cur_states, actions, rewards, next_states, users, action_index, valid_user_index, valid_feats
    # dim: cur_states,next_states = num_instances x num_features
    # dim: actions,rewards = num_instances

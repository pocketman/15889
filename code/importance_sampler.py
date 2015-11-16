import csv

PATH = 'C:\\Users\\REX\\Dropbox\\cmu\\fall2015\\15889\\project\\lectures\\'

def get_sample_policy(file_name):
    policy = get_action_counts(file_name)
    convert_action_counts_to_policy(policy)
    return policy

def get_action_counts(file_name):
    action_counts = dict([]);
    with open(file_name) as sample_file:
        file_reader = csv.DictReader(
                sample_file,
                fieldnames = ['user', 'action', 'reward'] ,
                restkey = 'state',
                delimiter = ',')
        state_values = next(file_reader, None)['state']
        for row in file_reader:
            # grab only the states which are 1s
            state = map(
                    lambda x, y: y if x == '1' else None,
                    row['state'],
                    state_values)
            state = filter(lambda x: x != None, state)

            state = frozenset(state)
            action = row['action']
            if state in action_counts:
                if action in action_counts[state]:
                    action_counts[state][action] = action_counts[state][action] + 1
                else:
                    action_counts[state][action] = 1
            else:
                action_counts[state] = {action: 1}
    return action_counts

def convert_action_counts_to_policy(action_counts):
    for actions in action_counts.values():
        actions_taken = reduce(lambda x, y: x + y, actions.values())
        for action in actions.keys():
            actions[action] = actions[action] * 1.0 / actions_taken

def policy_statistics(action_counts):
    action_tracker = dict([])
    for actions in policy.values():
        actions_taken = reduce(lambda x, y: x + y, actions.values())
        if actions_taken in action_tracker:
            action_tracker[actions_taken] = action_tracker[actions_taken] + 1
        else:
            action_tracker[actions_taken] = 1
    return action_tracker

def p_of_action_given_state(a, s, policy):
    actions = policy[s]
    if a in actions:
        return policy[a]
    else:
        return 0

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
            r_t = traj[t][2]
            p_test = p_of_action_given_state(a_t, s_t, test_policy)
            p_sample = p_of_action_given_state(a_t, s_t, sample_policy)
            p = p_test / p_sample * p
            utility += discount ** t * r_t * p
            t = t + 1
    return utility


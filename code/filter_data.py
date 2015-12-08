from collections import Counter
import csv

debug_state = ['0'] * 216
debug_state[120] = '1'
debug_state[-1] = '1'

def get_labels(labels_path):
    labels = []
    with open(labels_path) as labels_file:
        for line in labels_file:
            labels.append(line.strip())
    return labels

def is_valid_trajectory(trajectory, target_action):
    if len(trajectory) == 0:
        return False
    actions = map(lambda x: x['action'], trajectory)
    if not target_action in actions:
        return False
    actions_up_to_target = actions[:actions.index(target_action)]
    lectures = filter(lambda x: x[0] == 'L', actions_up_to_target)
    return len(lectures) > 0

def is_expert(trajectory):
    """
    Determines whether this trajectory belongs to an is_expert.
    We define an expert as someone who
    """
    return None


def trajectory_as_string(trajectory, target_action):
    """
    Only add events whose action are relevant to target_action
    """
    events = []
    for event in trajectory:
        if event['action'][0] == 'L' or event['action'] == target_action:
            l = [event['user'], event['action'], event['reward']]
            l.extend(event['next_state'])
            events.append(','.join(l))
    return '\n'.join(events) + '\n'

def filter_data(file_path, output_path, target_action, labels_path = 'labels.txt'):
    """
    filters out trajectories which have at most 1 lecture until target_action
    and filters out all quizzes not equal to target_action
    """
    file_out = open(output_path, 'w')
    with open(file_path) as file_pointer:
        file_reader = csv.DictReader(
                file_pointer,
                fieldnames = ['user', 'action', 'reward'] ,
                restkey = 'state',
                delimiter = ',')
        row = next(file_reader, None)
        user_label = row['user']
        action_label = row['action']
        reward_label = row['reward']
        state_values = row['state']
        state_labels = filter(lambda x: x[0] == 'L' or x == target_action, row['state'])
        labels = {'user': user_label, 'action': action_label, 'reward': reward_label, 'next_state': state_labels}
        file_out_labels = open(labels_path, 'w')
        file_out_labels.write('\n'.join(state_labels))
        file_out_labels.close()
        file_out.write(trajectory_as_string([labels]))
        previous_user = None
        trajectory_so_far = []
        for row in file_reader:
            state = map(lambda x, y: (x, y), row['state'], state_values)
            # filter out states features that arent lectures or target_action
            state = filter(lambda x: x[1][0] == 'L' or x[1] == target_action, state)
            state = map(lambda x: x[0], state)
            user = row['user']
            action = row['action']
            reward = row['reward']
            if (user != previous_user and is_valid_trajectory(trajectory_so_far, target_action)):
                for event in trajectory_so_far:
                    if event['next_state'] == debug_state:
                        print 'beginning of debug...'
                        print '\n'.join(map(lambda x: str(x), trajectory_so_far))
                        print 'end of debug-- is_valid: {a}'.format(
                            a = str(is_valid_trajectory(trajectory_so_far, target_action)))
                file_out.write(trajectory_as_string(trajectory_so_far, target_action))
                trajectory_so_far = []
            elif user != previous_user:
                #print str(map(lambda x: x['action'], trajectory_so_far))
                trajectory_so_far = []
            previous_user = user
            event = {'user': user, 'action': action, 'reward': reward, 'next_state': state}
            if not state_exists(event['next_state'], trajectory_so_far):
                trajectory_so_far.append(event)
        if is_valid_trajectory(trajectory_so_far, target_action):
            file_out.write(trajectory_as_string(trajectory_so_far, target_action))
    file_out.close()

def state_exists(state, trajectory):
    for event in trajectory:
        if state == event['next_state']:
            return True
    return False
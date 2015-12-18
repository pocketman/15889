from qlearn_policy import QLearnedPolicy
from load_data import iter_csv
import QLearning as ql
import numpy as np
from importance_sampler import estimate_utility, hcope
from collections import defaultdict
from load_data import read_feats
from sample_policy import SamplePolicy

q315_labels = ["L101","L103","L105","L107","L109","L11","L111","L113","L115","L117","L119","L121","L123","L125","L127","L129","L13","L131","L133","L135","L137","L141","L143","L147","L149","L15","L151","L153","L155","L157","L159","L161","L163","L165","L167","L169","L17","L171","L173","L175","L177","L179","L181","L183","L185","L187","L189","L19","L191","L193","L195","L197","L199","L201","L203","L205","L207","L209","L21","L211","L217","L219","L223","L225","L227","L23","L25","L27","L29","L31","L33","L35","L37","L39","L41","L43","L45","L47","L49","L5","L51","L53","L55","L57","L59","L61","L63","L65","L67","L69","L7","L71","L73","L75","L77","L79","L81","L83","L85","L87","L89","L9","L91","L93","L95","L97","L99","Q235","Q237","Q239","Q241","Q243","Q245","Q247","Q251","Q253","Q307","Q313","Q315","Q317","Q319","Q323","Q325","Q327","Q329","Q333","Q337","Q341","Q343","Q345","Q347","Q349","Q351","Q353","Q355","Q357","Q359","Q365","Q367","Q369","Q377"]
q315_labels = np.array(q315_labels)
discount = 0.99
delta = 0.05
q315_dir = 'C:\\Users\\REX\\Dropbox\\cmu\\fall2015\\15889\\project\\lectures\\q315'

action_q315 = np.where(q315_labels == 'Q315')[0][0]

states_315_train, actions_315_train, rewards_315_train, next_states_315_train, valid_actions_315_train, valid_users_315_train, trajectories_315_train = read_feats(q315_dir, 'train', action_q315)
states_315_test, actions_315_test, rewards_315_test, next_states_315_test, valid_actions_315_test, valid_users_315_test, trajectories_315_test = read_feats(q315_dir, 'test', action_q315)

states_315 = np.concatenate(
    (states_315_train, states_315_test),
    axis = 0)
next_states_315 = np.concatenate(
    (next_states_315_train, next_states_315_test),
    axis = 0)
actions_315 = np.concatenate(
    (actions_315_train, actions_315_test),
    axis = 0)
rewards_315 = np.concatenate(
    (rewards_315_train, rewards_315_test),
    axis = 0)

trajectories_315 = trajectories_315_train + trajectories_315_test

# evaluation for q315
print '-------------------Evaluation for Q315--------------------------------'
state_rewards_315 = ql.estimate_rewards(next_states_315_train, actions_315_train, rewards_315_train, action_q315)
discounted_rewards_315 = ql.discount_rewards(state_rewards_315, discount)
discounted_max_states_315 = ql.get_max_reward_states(discounted_rewards_315)
max_states_315 = ql.get_max_reward_states(state_rewards_315)
q315_policy = QLearnedPolicy(discounted_max_states_315[0], q315_labels)

print 'Reward of max state = {a}, discounted max state = {b}'.format(
    a = state_rewards_315[max_states_315[0]], b = state_rewards_315[discounted_max_states_315[0]])
print 'Discounted reward of max state = {a}, discounted max state = {b}'.format(
    a = discounted_rewards_315[max_states_315[0]], b = discounted_rewards_315[discounted_max_states_315[0]])
print 'Max state actions: {a} \nDiscounted max state actions: {b}'.format(
    a = q315_labels[np.array(max_states_315[0]).astype(int) == 1], b = q315_labels[np.array(discounted_max_states_315[0]).astype(int) == 1])

action_counts_315 = defaultdict(lambda: defaultdict(int))
for s,a in zip(states_315,actions_315):
    action_counts_315[tuple(s)][a] += 1
sample_policy_315 = SamplePolicy(action_counts_315)
expected_reward_315 = estimate_utility(sample_policy_315, q315_policy, trajectories_315_test, discount)
lower_bound_315 = hcope(sample_policy_315, q315_policy, trajectories_315_test, discount, delta)
print 'Expected Reward = {r}, Lower bound = {l}'.format(r = expected_reward_315, l = lower_bound_315)
print 'Expected Reward sample = {r}'.format(r = estimate_utility(sample_policy_315, sample_policy_315, trajectories_315_test, discount))
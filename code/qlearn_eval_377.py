from qlearn_policy import QLearnedPolicy
from load_data import iter_csv
import QLearning as ql
import numpy as np
from importance_sampler import estimate_utility, hcope
from load_data import read_feats
from collections import defaultdict
from sample_policy import SamplePolicy

q377_labels = ["L1","L101","L103","L105","L107","L109","L11","L111","L113","L115","L117","L119","L121","L123","L125","L127","L129","L13","L131","L133","L135","L137","L141","L143","L147","L149","L15","L151","L153","L155","L157","L159","L161","L163","L165","L167","L169","L17","L171","L173","L175","L177","L179","L181","L183","L185","L187","L189","L19","L191","L193","L195","L197","L199","L201","L203","L205","L207","L208","L209","L21","L211","L215","L217","L219","L223","L225","L226","L227","L23","L25","L27","L29","L31","L33","L35","L37","L39","L41","L43","L45","L47","L49","L5","L51","L53","L55","L57","L59","L61","L63","L65","L67","L69","L7","L71","L73","L75","L77","L79","L81","L83","L85","L87","L89","L9","L91","L93","L95","L97","L99","Q235","Q237","Q239","Q240","Q241","Q243","Q245","Q247","Q251","Q253","Q307","Q313","Q315","Q317","Q319","Q323","Q325","Q327","Q329","Q330","Q333","Q334","Q337","Q338","Q341","Q342","Q343","Q344","Q345","Q346","Q347","Q349","Q351","Q353","Q355","Q357","Q359","Q365","Q367","Q369","Q377"]
q377_labels = np.array(q377_labels)
discount = 0.99
delta = 0.05
q377_dir = 'C:\\Users\\REX\\Dropbox\\cmu\\fall2015\\15889\\project\\lectures\\q377'

action_q377 = np.where(q377_labels == 'Q377')[0][0]

states_377_train, actions_377_train, rewards_377_train, next_states_377_train, valid_actions_377_train, valid_users_377_train, trajectories_377_train = read_feats(q377_dir, 'train', action_q377)
states_377_test, actions_377_test, rewards_377_test, next_states_377_test, valid_actions_377_test, valid_users_377_test, trajectories_377_test = read_feats(q377_dir, 'test', action_q377)

states_377 = np.concatenate(
    (states_377_train, states_377_test),
    axis = 0)
next_states_377 = np.concatenate(
    (next_states_377_train, next_states_377_test),
    axis = 0)
actions_377 = np.concatenate(
    (actions_377_train, actions_377_test),
    axis = 0)
rewards_377 = np.concatenate(
    (rewards_377_train, rewards_377_test),
    axis = 0)

trajectories_377 = trajectories_377_train + trajectories_377_test

print '-------------------Evaluation for Q377--------------------------------'
# evaluation for q377
state_rewards_377 = ql.estimate_rewards(next_states_377_train, actions_377_train, rewards_377_train, action_q377)
discounted_rewards_377 = ql.discount_rewards(state_rewards_377, discount)
discounted_max_states_377 = ql.get_max_reward_states(discounted_rewards_377)
max_states_377 = ql.get_max_reward_states(state_rewards_377)
q377_policy = QLearnedPolicy(discounted_max_states_377[0], q377_labels)

print 'Reward of max state = {a}, discounted max state = {b}'.format(
    a = state_rewards_377[max_states_377[0]], b = state_rewards_377[discounted_max_states_377[0]])
print 'Discounted reward of max state = {a}, discounted max state = {b}'.format(
    a = discounted_rewards_377[max_states_377[0]], b = discounted_rewards_377[discounted_max_states_377[0]])
print 'Max state actions: {a} \nDiscounted max state actions: {b}'.format(
    a = q377_labels[np.array(max_states_377[0]).astype(int) == 1], b = q377_labels[np.array(discounted_max_states_377[0]).astype(int) == 1])

action_counts_377 = defaultdict(lambda: defaultdict(int))
for s,a in zip(states_377,actions_377):
    action_counts_377[tuple(s)][a] += 1
sample_policy_377 = SamplePolicy(action_counts_377)
expected_reward_377 = estimate_utility(sample_policy_377, q377_policy, trajectories_377_test, discount)
lower_bound_377 = hcope(sample_policy_377, q377_policy, trajectories_377_test, discount, delta)
print 'Expected Reward = {r}, Lower bound = {l}'.format(r = expected_reward_377, l = lower_bound_377)
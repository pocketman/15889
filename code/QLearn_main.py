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
target_action = "Q377"
discount = 0.99
labels = np.array(fd.get_labels(labels_path))
#feat_labels = np.array(["L1","L10","L100","L101","L102","L103","L104","L105","L106","L107","L108","L109","L11","L110","L111","L112","L113","L114","L115","L116","L117","L118","L119","L12","L120","L121","L122","L123","L124","L125","L126","L127","L128","L129","L13","L130","L131","L132","L133","L134","L135","L136","L137","L138","L14","L140","L141","L142","L143","L144","L147","L148","L149","L15","L150","L151","L152","L153","L154","L155","L156","L157","L158","L159","L16","L160","L161","L162","L163","L164","L165","L166","L167","L168","L169","L17","L170","L171","L172","L173","L174","L175","L176","L177","L178","L179","L18","L180","L181","L182","L183","L184","L185","L186","L187","L188","L189","L19","L190","L191","L192","L193","L194","L195","L196","L197","L198","L199","L2","L20","L200","L201","L202","L203","L204","L205","L206","L207","L208","L209","L21","L210","L211","L212","L215","L217","L219","L22","L220","L223","L225","L226","L227","L23","L24","L25","L26","L27","L28","L29","L30","L31","L32","L33","L34","L35","L36","L37","L38","L39","L40","L41","L42","L43","L44","L45","L46","L47","L48","L49","L5","L50","L51","L52","L53","L54","L55","L56","L57","L58","L59","L6","L60","L61","L62","L63","L64","L65","L66","L67","L68","L69","L7","L70","L71","L72","L73","L74","L75","L76","L77","L78","L79","L8","L80","L81","L82","L83","L84","L85","L86","L87","L88","L89","L9","L90","L91","L92","L93","L94","L95","L96","L97","L98","L99","Q-1","Q-2","Q1","Q2","Q235","Q237","Q239","Q240","Q241","Q243","Q245","Q247","Q251","Q253","Q3","Q307","Q308","Q313","Q314","Q315","Q316","Q317","Q318","Q319","Q320","Q323","Q324","Q325","Q327","Q329","Q330","Q333","Q334","Q337","Q338","Q341","Q342","Q343","Q344","Q345","Q346","Q347","Q348","Q349","Q350","Q351","Q352","Q353","Q354","Q355","Q356","Q357","Q358","Q359","Q360","Q365","Q366","Q367","Q368","Q369","Q370","Q377","Q379"])

cur_states, actions, rewards, next_states, users, action_index, user_index, valid_feats = load_data(
    feat_path,
    target_action,
    num_users_ratio = 1)
print cur_states.shape, actions.shape, rewards.shape, next_states.shape, len(user_index)

masked_feat_labels = []
for i in range(len(feat_labels)):
    if valid_feats[i]:
        masked_feat_labels.append(feat_labels[i])
masked_feat_labels = np.array(masked_feat_labels)
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
    a = masked_feat_labels[np.array(max_states[0]).astype(int) == 1], b = masked_feat_labels[np.array(discounted_max_states[0]).astype(int) == 1])
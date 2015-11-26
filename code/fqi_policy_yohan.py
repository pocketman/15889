from policy import Policy
import numpy as np


class FQIPolicy(Policy):

    def __init__(self, approximator, optimal_dist=False):
        self.approximator = approximator
        self.cache_p_actions_given_state = dict()  # #  p(a|s) = normalized distribution
        self.optimal_dist = optimal_dist

    def p_action_given_state(self, a, s):
        s_ = tuple(s)
        
        if self.cache_p_actions_given_state.has_key(s_):
            possible_actions = self.cache_p_actions_given_state[s_]
        else:
            possible_actions = self.approximator.predict([s_])  # n x 1 matrix
            possible_actions /= sum(possible_actions)
            self.cache_p_actions_given_state[s_] = possible_actions 
            
        if self.optimal_dist:
            return possible_actions[a,0]
        else:
            action = np.argmax(possible_actions)  # if there is tie, choose the first one
            if a == action: return 1
            else: return 0


    def get_action_given_state(self, s):
        s_ = tuple(s)
        
        if self.cache_p_actions_given_state.has_key(s_):
            possible_actions = self.cache_p_actions_given_state[s_]
        else:
            possible_actions = self.approximator.predict([s_])
            possible_actions /= sum(possible_actions)
            self.cache_p_actions_given_state[s_] = possible_actions 
        
        if self.optimal_dist:
            return np.random.choice(len(possible_actions), p=possible_actions)
        else:
            return np.argmax(possible_actions)  # if there is tie, choose the first one

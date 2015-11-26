from policy import Policy
import numpy as np


class FQIPolicy(Policy):

    def __init__(self, approximator, top_k=-1):
        self.approximator = approximator
        self.cache_p_actions_given_state = dict()  # #  p(a|s) = normalized distribution
        self.top_k = top_k

    def p_action_given_state(self, a, s):
        s_ = tuple(s)
        
        if self.cache_p_actions_given_state.has_key(s_):
            return self.cache_p_actions_given_state[s_][a]
        
        possible_actions = self.approximator.predict([s_])  # n x 1 matrix
            
        if self.top_k == 1:
            max_a = np.argmax(possible_actions)  # if there is tie, choose the first one
            possible_actions_ = np.zeros(len(possible_actions))
            possible_actions_[max_a] = 1
            self.cache_p_actions_given_state[s_] = possible_actions_
        else:
            possible_actions_ = possible_actions.flatten()
            if self.top_k != -1:
                top_indices = np.argpartition(-possible_actions_, self.top_k)
                possible_actions_[top_indices[self.top_k:]] = 0
            possible_actions_ /= sum(possible_actions_)
            self.cache_p_actions_given_state[s_] = possible_actions_
        return possible_actions_[a]
            

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

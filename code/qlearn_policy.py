from policy import policy
import numpy as np

class QLearnPolicy(Policy):

    def __init__(self, max_state):
        self.max_states = max_state

    def p_action_given_state(self, a, s):
        if a in self.max_states and s[a] != 1:
            return 1.0 / sum(s)
        else:
            return 0

    def get_action_given_state(self, s):
        raise Exception('Not implemented')
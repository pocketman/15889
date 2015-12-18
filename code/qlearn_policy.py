from policy import Policy
import numpy as np

class QLearnedPolicy(Policy):
    def __init__(self, best_state, state_labels, other_prob = 0.1):
        self.best_state = best_state
        self.state_labels = state_labels
        self.other_prob = other_prob

    def p_action_given_state(self, a, s):
        if self.best_state[a] == 1 and s[a] == 0:
            return 1 - self.other_prob
        else:
            print self.other_prob / (len(s) - 1)
            return self.other_prob / (len(s) - 1)

    def get_action_given_state(self, s):
        raise Exception('Not implemented')
        return None


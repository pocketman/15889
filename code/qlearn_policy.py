from policy import Policy
import numpy as np

class QLearnedPolicy(Policy):
    def __init__(self, best_state, state_labels, other_prob = 0.1):
        self.best_state = np.array(best_state)
        self.state_labels = state_labels
        self.other_prob = other_prob

    def p_action_given_state(self, a, s):
        actions_to_take = self.actions_to_take(s)
        if a in actions_to_take:
            # haven't taken this action but it is in our interest to
            return (1 - self.other_prob) / len(actions_to_take)
        else:
            return self.other_prob / (len(np.where(np.array(s) == 0)[0]) - len(actions_to_take))

    def actions_to_take(self, s):
        """
        s - state we are in
        returns list of actions we should still take given the state
        """
        actions_in_best_state = np.where(self.best_state == 1)[0]
        actions_to_take = []
        for a in actions_in_best_state:
            if s[a] == 0:
                actions_to_take.append(a)
        return np.array(actions_to_take)

    def get_action_given_state(self, s):
        raise Exception('Not implemented')
        return None


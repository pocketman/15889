from policy import Policy
import random as rng

class SamplePolicy(Policy):

    def __init__(self, action_counts):
        self.action_counts = action_counts  # dict[state][action] = int
        self.cache_p_action_given_state = dict()
        

    def p_action_given_state(self, a, s):
        """
        gets the probability of taking action a in state s_
        """
        s_ = tuple(s)
        if self.cache_p_action_given_state.has_key((s_,a)):
            return self.cache_p_action_given_state[(s_,a)]
        
        if s_ in self.action_counts:
            possible_actions = self.action_counts[s_]
            if a in possible_actions:
                actions_taken = reduce(
                        lambda x, y: x + y, possible_actions.values())
                p = possible_actions[a] / float(actions_taken)
                self.cache_p_action_given_state[(s_,a)] = p
                return p
            else:
                return 0
        else:
            raise Exception(
                    'Sample Policy does not have any data for action {act}!'
                            .format(act = a))


    def get_action_given_state(self, s):
        """
        Returns the action to take given we are in state s_.
        Since sample policy is stochastic, then our returned action is random
        """
        s_ = tuple(s)
        
        if s_ in self.action_counts:
            possible_actions = self.action_counts[s_]
            actions_taken = reduce(
                    lambda x, y: x + y, possible_actions.values())
            i = rng.randint(1, actions_taken)
            action_names = possible_actions.keys()
            curr_action = action_names.pop(0)
            while i > 1 and i > possible_actions[curr_action]:
                i = i - possible_actions[curr_action]
                curr_action = action_names.pop(0)
            return curr_action
        else:
            raise Exception(
                    'Sample Policy does not have any data for state!')

    def policy_statistics(self):
        """
        Counter of the number of state occurrences
        """
        action_tracker = dict([])
        for actions in self.action_counts.values():
            actions_taken = reduce(lambda x, y: x + y, actions.values())
            if actions_taken in action_tracker:
                action_tracker[actions_taken] = action_tracker[actions_taken] + 1
            else:
                action_tracker[actions_taken] = 1
        return action_tracker

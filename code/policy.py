from abc import ABCMeta, abstractmethod

class Policy:
    """
    All policies should implement this interface
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def p_action_given_state(self, a, s):
        """
        should return the probability of selecting a given state s
        """
        return 0

    @abstractmethod
    def get_action_given_state(self, s):
        """
        should return the action which maximizes reward given state s
        """
        return 0

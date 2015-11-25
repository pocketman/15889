from policy import Policy
import numpy as np

class FQIPolicy(Policy):

    def __init__(self, approximator, feature_list, valid_features, valid_actions):
        self.approximator = approximator
        # filter by valid features
        features_to_keep = map(lambda x, y: x if y == '1' else None, feature_list, valid_features)
        features_to_keep = filter(lambda x: x != None, features_to_keep)

        self.feature_mapper = dict()
        # map each feature to the index it would be in the input state vector
        for i in range(len(features_to_keep)):
            self.feature_mapper[features_to_keep[i]] = i
        self.valid_actions = valid_actions

    def p_action_given_state(self, a, s):
        state = np.zeros((1, len(self.feature_mapper)))
        for feat in s:
            if feat in self.feature_mapper:
                state[0][self.feature_mapper[feat]] = 1
        possible_actions = self.approximator.predict(state)
        action = self.valid_actions[np.argmax(possible_actions)]
        if a == action:
            return 1
        else:
            return 0

    def get_action_given_state(self, s):
        state = np.zeros((1, len(self.feature_mapper)))
        for feat in s:
            if feat in self.feature_mapper:
                state[0][self.feature_mapper[feat]] = 1
        possible_actions = self.approximator.predict(state)
        action = self.valid_actions[np.argmax(possible_actions)]
        return action
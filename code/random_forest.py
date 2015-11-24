'''
Created on Nov 23, 2015

@author: Yohan
'''
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
import numpy as np

class RandomForest():
    '''
    classdocs
    '''

    def __init__(self, num_estimators, num_actions):
        '''
        Constructor
        '''
        self.estimators_num = num_estimators
        self.num_actions = num_actions
        self.clfs = [ RandomForestRegressor(n_estimators=num_estimators, max_depth=10) for a in range(num_actions) ] 
    
    
    def train(self, xs, actions, ys):
        # xs,ys = num_instances x num_features
        # actions = num_instances x 1
        for a in range(self.num_actions):
            self.clfs[a].fit(xs[actions==a], ys[actions==a])
    
    def predict(self, train_xs):
        res = np.empty((self.num_actions, len(train_xs)))
#         print "!!", len(train_xs)
        for a in range(self.num_actions):
            res[a,:] = self.clfs[a].predict(train_xs)
        return res
    
    def save(self, path):
        joblib.dump(self, path)
        # to load, call: rf = joblib.load(path)


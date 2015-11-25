from importance_sampler import *
import convert_to_policy as ctp
import numpy as np


PATH = 'C:\\Users\\REX\\Dropbox\\cmu\\fall2015\\15889\\project\\lectures\\'
policy = get_sample_policy(PATH + 'feats_test.csv')
forest_dir = 'd0.9-u12000-e10-i1000-tQ315'
test_policy = ctp.convert_random_forest_to_policy(forest_dir)
trajectories = load_trajectories(PATH + 'feats.csv')
sample_policy = get_sample_policy(PATH + 'feats.csv')
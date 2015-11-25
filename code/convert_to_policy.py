from sklearn.externals import joblib
from fqi_policy import FQIPolicy
import csv


DATA_DIR = 'C:\\Users\\REX\\Dropbox\\cmu\\fall2015\\15889\\project\\lectures\\'
RESULTS_DIR = 'C:\\Users\\REX\\Dropbox\\cmu\\fall2015\\15889\\project\\results\\'

def get_random_forest(dir_name):
    return joblib.load(RESULTS_DIR + dir_name + '\\approximator\\random_forest_regressor.model');


def convert_random_forest_to_policy(dir_name):
    random_forest = get_random_forest(dir_name)
    valid_feats = []
    valid_actions = []

    with open(RESULTS_DIR + dir_name + '\\valid_feats.txt') as feat_file:
        for row in feat_file:
            valid_feats.append(row.strip())

    with open(RESULTS_DIR + dir_name + '\\valid_actions.txt') as actions_file:
        for row in actions_file:
            valid_actions.append(row.strip())

    with open(DATA_DIR + 'feats.csv') as data_file:
        file_reader = csv.DictReader(
                data_file,
                fieldnames = ['user', 'action', 'reward'] ,
                restkey = 'state',
                delimiter = ',')
        features_list = next(file_reader, None)['state']

    return FQIPolicy(random_forest, features_list, valid_feats, valid_actions)


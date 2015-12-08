import filter_data as fd
from importance_sampler import *

PATH = "C:\\Users\\REX\\Dropbox\\cmu\\fall2015\\15889\\project\\lectures\\feats.csv"
OUT_PATH = "C:\\Users\\REX\\Dropbox\\cmu\\fall2015\\15889\\project\\lectures\\filtered_feats.csv"
labels_path = 'C:\\Users\\REX\\Dropbox\\cmu\\fall2015\\15889\\project\\lectures\\filtered_feats_labels.csv'
target_action = 'Q315'
fd.filter_data(PATH, OUT_PATH, target_action, labels_path = labels_path)
'''
Created on Nov 11, 2015
For RL Project

@author: Yohan
'''

import re
from collections import defaultdict
from coursera.model.clickstream import Clickstream
from coursera.model.quiz import Quiz
from coursera.model.lecture import Lecture
import csv
import sys

# in_path = "/Users/Yohan/Dropbox/Research/data/MOOCs/algebra-001/algebra-001_clickstream_export"
db_path = "/Users/Yohan/Dropbox/Research/data/MOOCs/algebra-001/Intermediate Algebra (algebra-001)_SQL_anonymized_general.sql"
demo_path = "/Users/Yohan/Dropbox/Research/data/MOOCs/algebra-001/algebra-001_Demographics_individual_responses.csv"
out_dir = "/Users/Yohan/Dropbox/CMU/Semesters/2015 Fall/15889 RL/pj/data/lectures+demography"
seqs_path = out_dir+"/user_seqs.txt"
seq_len_path = out_dir+"/seqs_length.csv"
action_path = out_dir+"/actions.csv"
feat_path = out_dir+"/feats.csv"

demo_csv = csv.reader(open(demo_path))
demo_csv.next()
raw_demo_header = demo_csv.next()
demo_header = []
demo_start, demo_end = (20,56)
for h in raw_demo_header[demo_start:(demo_end+1)]:
    demo_header.append(re.sub(".*-", "", h).strip())
print demo_header
demo_map = dict()
for row in demo_csv:
    demo_map[row[2]] = [ 1 if r=="TRUE" else 0 for r in row[demo_start:(demo_end+1)] ]


action_cnt = defaultdict(int)
user_seqs = defaultdict(list)
for lect in Lecture.lecture_submissions(db_path):
    user_seqs[lect['session_user_id']].append((lect['submission_time'],"L"+lect['item_id'],"0"))  # include both 'view' and 'download'
    action_cnt["L"+lect['item_id']] += 1
    
for quiz in Quiz.quiz_submissions(db_path):
    user_seqs[quiz['session_user_id']].append((quiz['submission_time'],"Q"+quiz['item_id'],(0 if quiz['raw_score']=="NULL" else quiz['raw_score'])))
    action_cnt["Q"+quiz['item_id']] += 1


# quiz_submission = dict()
# for log in Clickstream.logs(in_path):
#     action = re.sub("/$","", re.sub("https://class.coursera.org/algebra-001/", "", log["page_url"]))
#     
#     if not re.search("^(lecture|signature|quiz)", action): continue
#     action_refined = None
#     
#     # lecture
#     if re.search("view\\?lecture_id=([\\d]+)$", action):
#         action_refined = re.sub("view\\?lecture_id=([\\d]+)", "\\1", action)
#     elif re.search("^lecture.*/([\\d]+)$", action):
#         action_refined = re.sub("^lecture.*/([\\d]+)", "lecture/\\1", action)
#     
#     # quiz
#     if re.search("signature/modal_iframe",action) or re.search("feedback\\?submission_id",action):
#         m = re.search("signature/modal_iframe\\?submission_id=([\\d]+)&type=quiz", action)
#         if m == None: m = re.search("feedback\\?submission_id=([\\d]+)", action)
#         if m == None: continue
#         action_refined = "quiz/submission/"+quiz_submission[m.group(1)]
#     
#     
#     '''
#     if action.startswith("wiki/edit"): action = "wiki/edit"
#     elif action.startswith("wiki/view"): action = "wiki/view"
#     
#     if action == "class" or action.startswith("class/index"): action = "class/index"
#     elif action.startswith("class/preferences"): action = "class/preferences"
# 
#     action = re.sub("auth/welcome\\?.*", "auth/welcome", action)
#     if action.startswith("auth/login_receiver?"): action = "auth/login_receiver"
#     action = re.sub("auth/stop_emails\\?.*", "auth/stop_emails", action)
#     
#     
#     action = re.sub("forum/thread\\?.*(thread_id=[\\d]+).*", "forum/thread?\\1", action)
#     action = re.sub("forum/search\\?.*", "forum/search", action)
#     if action=="forum" or action.startswith("forum/index"): action = "forum/index"
#     elif action.startswith("forum/list"): action = "forum/list"
#     elif action.startswith("forum/profile?user_id"): action = "forum/profile?user_id"
#     elif action.startswith("forum/tag"): action = "forum/tag"
#     elif action.startswith("forum/thread"): action = "forum/thread"
#         
#     if action.startswith("generic/apply_late_days?item_type=quiz"): action = "generic/apply_late_days?item_type=quiz"
#     '''
# 
#     if action_refined == None: continue
# 
#     user_seqs[log["username"]].append((log["timestamp"],action_refined))
#     action_cnt[action_refined] += 1



# Sort and remove same actions in a row
for user,seq in user_seqs.iteritems():
    seq_refined = []
    prev_action = None
    for action in sorted(seq):  # same action in a row
        if action[1]==prev_action: continue
        seq_refined.append(action)
        prev_action = action[1]
    user_seqs[user] = seq_refined
    

# print seqs
len_seqs = defaultdict(int)
seqs_file = open(seqs_path,"w")
for user,seqs in user_seqs.iteritems():
    seqs_file.write(user)
    for action in [ a for t,a,r in sorted(seqs) ]:
        seqs_file.write("\t"+action)
    seqs_file.write("\n")
    len_seqs[len(seqs)] += 1
seqs_file.close()


# print features
action_index = dict()
for i,(action,cnt) in enumerate(sorted(action_cnt.iteritems())):
    action_index[action] = i
    
feat_file = open(feat_path,"w")
out_csv = csv.writer(feat_file)
out_csv.writerow(["user","action","reward","demo_unavailable"]+demo_header+sorted(action_index.keys()))
for user,seq in user_seqs.iteritems():
    feat = [ 0 for train_x in range(len(action_cnt)) ]
    for timestamp,action,reward in seq:
        feat[action_index[action]] = 1  # binary
        if demo_map.has_key(user): demo_feat = [0] + demo_map[user]
        else: demo_feat = [1] + [ 0 for d in demo_header ] 
        out_csv.writerow([user,action,reward]+demo_feat+feat)
feat_file.close()


# print actions
action_file = open(action_path,"w")
for action,cnt in sorted(action_cnt.iteritems()):
    action_file.write(action+","+str(cnt)+"\n")
action_file.close()


# seq len
sum_len_seqs = sum([k*v for k,v in len_seqs.iteritems()])
num_seqs = sum(len_seqs.values())
seq_len_file = open(seq_len_path,"w")
sum_cnt = 0
for length,cnt in sorted(len_seqs.iteritems()):
    seq_len_file.write(str(length)+","+str(cnt)+"\n")
    if sum_cnt < num_seqs / 2 <= sum_cnt+cnt: median = length
    sum_cnt += cnt
seq_len_file.write("SumLenSeqs,"+str(sum_len_seqs)+"\n")
seq_len_file.write("NumSeqs,"+str(num_seqs)+"\n")
seq_len_file.write("Avg,"+str(1.0*sum_len_seqs/num_seqs)+"\n")
seq_len_file.write("Median,"+str(median)+"\n")
seq_len_file.close()



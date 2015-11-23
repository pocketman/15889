'''
Created on Jun 23, 2015

@author: Yohan
'''
from collections import defaultdict
import csv
import json
import re
import sys
from model.lecture import Lecture



class Clickstream:

    def __init__(self, path):
        pass

    @staticmethod
    def logs(path):
        cnt = 0
        for line in open(path):
            cnt += 1
            if cnt % 10000 == 0: sys.stderr.write('.')
            if cnt % 100000 == 0: sys.stderr.write('(%d)' % cnt)
            if cnt % 500000 == 0: sys.stderr.write('\n')
            yield json.loads(line)
        sys.stderr.write('\n')

    @staticmethod
    def write_user_first_last_access(click_path, outpath_hour):
        user_first_access = defaultdict(lambda: sys.maxint)
        user_last_access = defaultdict(lambda: 0)
        for log in Clickstream.logs(click_path):
            if user_first_access[log['username']] > log['timestamp']/1000:
                user_first_access[log['username']] = log['timestamp']/1000
            if user_last_access[log['username']] < log['timestamp']/1000:
                user_last_access[log['username']] = log['timestamp']/1000
        
        outfile = open(outpath_hour,'w')
        out_csv = csv.writer(outfile)
        out_csv.writerow(['username','first_access','last_access'])
        out_csv.writerows( [ [username, user_first_access[username], user_last_access[username]] for username in user_first_access.keys() ] )
        outfile.close()

    @staticmethod
    def load_user_first_last_access(path):
        user_first_access = dict()
        user_last_access = dict()
        infile = open(path)
        in_csv = csv.reader(infile)
        in_csv.next()  # header
        for row in in_csv:
            user_first_access[row[0]] = int(row[1])
            user_last_access[row[0]] = int(row[2])
        return user_first_access, user_last_access

    @staticmethod
    def write_dropout_time(click_path, lect_path, outpath_hour):
        lectures = [ lect for lect in Lecture.lectures(lect_path) if Lecture.is_valid_lecture(lect) ]
        lectures.sort(key=lambda l: int(l['open_time']))
        second_half_lects = set([ l['id'] for l in lectures[((len(lectures)+1)/2):] ])

        user_lects = defaultdict(set)
        user_last_access = defaultdict(int)    
        for log in Clickstream.logs(click_path):
            if log['timestamp']/1000 > user_last_access[log['username']]:
                user_last_access[log['username']] = log['timestamp']/1000
            
            m = re.search('lecture_id=([\\d]+)', log['page_url'])
            if m==None: m = re.search('lecture/([\\d]+)', log['page_url'])
            if m == None: continue
            user_lects[log['username']].add(m.group(1))
        
        dropout_time = dict()
        for username, lects in user_lects.iteritems():
            if len( lects & second_half_lects )==0:
                dropout_time[username] = user_last_access[username]
        
        outfile = open(outpath_hour,'w')
        out_csv = csv.writer(outfile)
        out_csv.writerow(['username','dropout_time'])
        out_csv.writerows( [ [username, time] for username,time in dropout_time.iteritems() ] )
        outfile.close()

    @staticmethod
    def load_dropout_time(path):
        dropout_time = dict()
        infile = open(path)
        in_csv = csv.reader(infile)
        in_csv.next()  # header
        for row in in_csv:
            dropout_time[row[0]] = int(row[1])
        return dropout_time
        
    
    
# click_path = '/Users/Yohan/Research/data/MOOCs/algebra-001/algebra-001_clickstream_export'
# db_path = '/Users/Yohan/Research/data/MOOCs/algebra-001/Intermediate Algebra (algebra-001)_SQL_anonymized_general.sql'
# first_last_access_path = '/Users/Yohan/Dropbox/edX/data/Algebra-001/first_last_access.csv'
# dropout_path = '/Users/Yohan/Dropbox/edX/data/Algebra-001/dropout_time.csv'
# outpath_hour = "/Users/Yohan/Dropbox/edX/data/Algebra-001/lag.csv"

# Write dropout_time
#Clickstream.write_dropout_time(click_path, db_path, dropout_path)



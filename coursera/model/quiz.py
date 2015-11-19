'''
Created on Jul 7, 2015

@author: Yohan
'''
import csv
import re
from collections import defaultdict

class Quiz:

    def __init__(self, params):
        pass
    
    
    @staticmethod
    def quizzes(db_path):
        fields = ['id','parent_id','open_time','soft_close_time','hard_close_time','maximum_submissions','title','duration','quiz_type','proctoring_requirement','authentication_required','deleted','last_updated']
        for line in open(db_path):
            if not line.startswith("INSERT INTO `quiz_metadata` VALUES "): continue
            for item in csv.DictReader( map(lambda s: re.sub('^\\(','', re.sub('\\);?$','',s)), re.split('(?<=\\)),(?=\\()', line.replace("INSERT INTO `quiz_metadata` VALUES ",""))), quotechar="'", fieldnames=fields ):
                yield item


    @staticmethod
    def quiz_submissions(db_path):
        fields = ['id','item_id','session_user_id','submission_time','submission_number','raw_score','grading_error','authenticated_submission_id']
        for line in open(db_path):
            if not line.startswith("INSERT INTO `quiz_submission_metadata` VALUES "): continue
            for item in csv.DictReader( map(lambda s: re.sub('^\\(','', re.sub('\\);?$','',s)), re.split('(?<=\\)),(?=\\()', line.replace("INSERT INTO `quiz_submission_metadata` VALUES ",""))), quotechar="'", fieldnames=fields ):
                yield item

 
    @staticmethod
    def valid_quizzes(db_path, threshold):
        quiz_users = defaultdict(set)
        for submission in Quiz.quiz_submissions(db_path):
            quiz_users[submission['item_id']].add(submission['session_user_id'])
        return set( [quiz_id for quiz_id,users in quiz_users.iteritems() if len(users) >= threshold] )

    
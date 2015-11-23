'''
Created on Nov 5, 2015

@author: Yohan
'''
import csv
import re

class Assignment:
    '''
    classdocs
    '''


    def __init__(self, params):
        '''
        Constructor
        '''
        
    @staticmethod
    def assignments(db_path):
        fields = ['id','parent_id','open_time','soft_close_time','hard_close_time','title','maximum_submissions','deleted','last_updated']
        for line in open(db_path):
            if not line.startswith("INSERT INTO `assignment_metadata` VALUES "): continue
            for item in csv.DictReader( map(lambda s: re.sub('^\\(','', re.sub('\\);?$','',s)), re.split('(?<=\\)),(?=\\()', line.replace("INSERT INTO `assignment_metadata` VALUES ",""))), quotechar="'", fieldnames=fields ):
                yield item

    @staticmethod
    def assignment_submissions(db_path):
        fields = ['id','item_id','session_user_id','submission_time','submission_number','raw_score','authenticated_submission_id']
        for line in open(db_path):
            if not line.startswith("INSERT INTO `assignment_submission_metadata` VALUES "): continue
            for item in csv.DictReader( map(lambda s: re.sub('^\\(','', re.sub('\\);?$','',s)), re.split('(?<=\\)),(?=\\()', line.replace("INSERT INTO `assignment_submission_metadata` VALUES ",""))), quotechar="'", fieldnames=fields ):
                yield item

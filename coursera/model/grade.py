'''
Created on Jul 5, 2015

@author: Yohan
'''
import csv
import re


class Grades(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        self.fields = ['id','session_user_id','normal_grade','distinction_grade','achievement_level','authenticated_overall','ace_grade','passed_ace']
        self.data = dict()  # id : {row}


    @staticmethod
    def grades(path):
        fields = ['id','session_user_id','normal_grade','distinction_grade','achievement_level','authenticated_overall','ace_grade','passed_ace']
        for line in open(path):
            if not line.startswith("INSERT INTO `course_grades` VALUES "): continue
            for item in csv.DictReader( map(lambda s: re.sub('^\\(','', re.sub('\\);?$','',s)), re.split('(?<=\\)),(?=\\()', line.replace("INSERT INTO `course_grades` VALUES ",""))), quotechar="'", fieldnames=fields ):
                yield item

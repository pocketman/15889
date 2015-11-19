'''
Created on Jun 22, 2015

@author: Yohan
'''
import re
import csv
import time
import sys

class Lecture:

    # timestamps are 10 digits
    def __init__(self, path):
        self.fields = ['id','parent_id','open_time','soft_close_time','hard_close_time','maximum_submissions','title','source_video','video_length','quiz_id','final','deleted','last_updated','video_id','video_id_v2']
        self.data = dict()  # id : {row}
        self.loadLectures(path)
        
    
    @staticmethod
    def lectures(db_path):
        fields = ['id','parent_id','open_time','soft_close_time','hard_close_time','maximum_submissions','title','source_video','video_length','quiz_id','final','deleted','last_updated','video_id','video_id_v2']
        for line in open(db_path):
            if line.startswith("INSERT INTO `lecture_metadata` VALUES "): break
        for item in csv.DictReader( map(lambda s: re.sub('^\\(','', re.sub('\\);?$','',s)), re.split('(?<=\\)),(?=\\()', line.replace("INSERT INTO `lecture_metadata` VALUES ",""))), quotechar="'", fieldnames=fields ):
            item['open_gmtime'] = ( 'NULL' if item['open_time']=='NULL' else time.strftime("%Y.%m.%d %H:%M:%S (%w)",time.gmtime(int(item['open_time']))) )
            yield item
   
    @staticmethod
    def lecture_submissions(db_path):
        fields = ['id','item_id','session_user_id','submission_time','submission_number','raw_score','action']
        for line in open(db_path):
            if not line.startswith("INSERT INTO `lecture_submission_metadata` VALUES "): continue
            for item in csv.DictReader( map(lambda s: re.sub('^\\(','', re.sub('\\);?$','',s)), re.split('(?<=\\)),(?=\\()', line.replace("INSERT INTO `lecture_submission_metadata` VALUES ",""))), quotechar="'", fieldnames=fields ):
                yield item
   
    
    @staticmethod
    def lecture_norm_ids(path):
        norm = dict()
        for lecture in sorted([ l for l in Lecture.lectures(path) ], key=lambda l: int(l['parent_id'])):
            if lecture['parent_id'] == '-1': norm[lecture['id']] = lecture['id']
            else: norm[lecture['id']] = norm[lecture['parent_id']]
        return norm
    
    @staticmethod
    def get_first_lecture_time(path):
        first_lect_time = sys.maxint
        for l in Lecture.lectures(path):
            if Lecture.is_valid_lecture(l) and int(l['open_time']) < first_lect_time:
                first_lect_time = int(l['open_time'])
        return first_lect_time

    @staticmethod
    def is_valid_lecture(lect):
        return lect['open_time'] != 'NULL' and lect['source_video'] != 'NULL' and lect['soft_close_time'] != '0'

    
    def loadLectures(self, path):
        for lecture in Lecture.lectures(path):
            self.data[lecture['id']] = lecture
    
    
    def writeCSV(self, path):
        header = self.fields[:3]+['open_gmtime']+self.fields[3:]
        outfile = open(path,'w')
        out_csv = csv.writer(outfile)
        out_csv.writerow(header)
        out_csv.writerows( [ [ row[field] for field in header ] for row in sorted(self.data.values(), key=lambda v: int(v['id'])) ] )
        outfile.close()

    

# l = Lecture('/Users/Yohan/Research/data/MOOCs/algebra-001/Intermediate Algebra (algebra-001)_SQL_anonymized_general.sql')
# l.writeCSV('/Users/Yohan/Dropbox/edX/data/Algebra-001/all_lectures.csv')

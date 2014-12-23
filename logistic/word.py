#coding=utf-8

from math import log

class word(object):
    def __init__(self,word_name,word_no):
        self.word_name = word_name
        self.word_no = word_no
        self.word_tf = 0.
        self.word_idf = 0.
        self.word_docs = 0
        self.word_shang = 0.
        self.word_cat_dict = {}

    def update_dict(self,cat_str):
        self.word_docs += 1
        if cat_str in self.word_cat_dict:
            self.word_cat_dict[cat_str] += 1
        else:
            self.word_cat_dict[cat_str] = 1

    def update_tf(self):
        self.word_tf += 1
    def reset_tf(self):
        self.word_tf = 0
    def get_tf(self,wordnum):
        self.word_tf = 1.*self.word_tf / wordnum
        return self.word_tf
    
    def get_idf(self,docnum):
        self.word_idf = log(1.*docnum / self.word_docs)
        return self.word_idf

    def get_tfidf(self):
        return self.word_tf * self.word_idf

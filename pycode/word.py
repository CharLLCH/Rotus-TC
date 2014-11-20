#coding=utf-8
#one word type to get more info.
from math import log

class word:
    def __init__(self,w_name,w_idf = 0,w_docs = 0,w_s = 0,w_cat_dict = {}):
        self.__w_name = w_name
        self.__w_idf = w_idf
        self.__w_docs = 0
        self.__w_s = 0
        self.__w_cat_dict = w_cat_dict

    def get_s(self,doc_num,num_set):
		H = -10*(1.0/10)*log(1.0/10,2)
		p1 = self.__w_docs*1.0 / doc_num
		p2 = 1 - p1
		H1 = 0.0
		H2 = 0.0
		for idx in self.__w_cat_dict:
			p11 = self.__w_cat_dict[idx]*1.0 / self.__w_docs
			H1 += p11*log(p11,2)
			p22 = (num_set[idx]-self.__w_cat_dict[idx])*1.0 / (doc_num-self.__w_docs)
			H2 += p22*log(p22,2)
		self.__w_s = H + p1*H1 + p2*H2
		return self.__w_s

    def get_svalue(self):
        return self.__w_s

    def get_docs_num(self):
        return self.__w_docs

    def update_dict(self,cat_str):
        if cat_str in self.__w_cat_dict:
            self.__w_cat_dict[cat_str] += 1
        else:
            self.__w_cat_dict[cat_str] = 1

    def get_wname(self):
        return self.__w_name

    def get_widf(self,doc_num):
		self.__w_idf = log(doc_num*1.0 / self.__w_docs)
		return self.__w_idf
    
    def get_wcatdict(self):
        return self.__w_cat_dict

    def get_docs(self):
        for idx in self.__w_cat_dict:
            self.__w_docs += self.__w_cat_dict[idx]


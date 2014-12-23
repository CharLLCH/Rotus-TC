#coding=utf-8

from word import word
from read_conf import config
from nlp import NLP
import numpy as np
import os
from sklearn import linear_model
from logistic_nd import LogisticRegression

data_conf = config('../conf/dp.conf')
tr_data_path = data_conf['train_path']
te_data_path = data_conf['test_path']

cat_dict = {'acq':0,'corn':1,'crude':2,'earn':3,'grain':4,'interest':5,'money-fx':6,'ship':7,'trade':8,'wheat':9}
nlp = NLP()

def get_doc_num(path):
    docs_dict = {'doc_sum':0}
    doc_dir = os.listdir(path)
    for doc_cat in doc_dir:
        file_list = os.listdir(path+doc_cat)
        docs_dict[cat_dict[doc_cat]] = len(file_list)
        docs_dict['doc_sum'] += docs_dict[cat_dict[doc_cat]]
    return docs_dict

def get_voc_set():
    word_dict = {}
    word_no = 0
    doc_dir = os.listdir(tr_data_path)
    for doc_cat in doc_dir:
        file_list = os.listdir(tr_data_path+doc_cat)
        print '开始处理： '+doc_cat+' 文件夹文件'
        for file_path in file_list:
            doc_f = open(tr_data_path+doc_cat+'/'+file_path,'rb')
            document = doc_f.read()
            tokens = set(nlp.word_tokenize(document))
            for w in tokens:
                if w not in word_dict:
                    word_dict[w] = word(w,word_no)
                    word_no += 1
                word_dict[w].update_dict(cat_dict[doc_cat])
    return word_dict

def get_voc_vector(word_dict,doc_path,feattype,self_logstic):
    D = len(word_dict)
    docs_dict = get_doc_num(doc_path)
    N = docs_dict['doc_sum']
    x = np.matrix([0.]*N*D).reshape(N,D)
    y_cat = []
    doc_idx = 0
    doc_dir = os.listdir(doc_path)
    for doc_cat in doc_dir:
        f_list = os.listdir(doc_path+doc_cat)
        print '转换： '+doc_cat+' 文件夹文件'
        for f_p in f_list:
            y_cat.append(cat_dict[doc_cat])
            doc_f = open(doc_path+doc_cat+'/'+f_p,'rb')
            document = doc_f.read()
            tokens = nlp.word_tokenize(document)
            for w in tokens:
                if w in word_dict:
                    word_dict[w].update_tf()
            for w in set(tokens):
                if w in word_dict:
                    word_dict[w].get_tf(len(tokens))
                    word_dict[w].get_idf(N)
                    if feattype == 0:
                        x[doc_idx,word_dict[w].word_no] = word_dict[w].get_tfidf()
                    elif feattype == 1:
                        x[doc_idx,word_dict[w].word_no] = word_dict[w].word_idf
                    else:
                        x[doc_idx,word_dict[w].word_no] = word_dict[w].word_tf
                    word_dict[w].reset_tf()
            doc_idx += 1
    if self_logstic == True:
        C = len(cat_dict)
        y = np.matrix([0]*N*C).reshape(N,C)
        for idx in xrange(N):
            y[idx,y_cat[idx]] = 1
    else:
        y = np.array(y_cat)
    return x,y

def get_pred(pred):
    y_list = []
    for n_idx in xrange(pred.shape[0]):
        tmp_max = 0
        tmp_idx = 0
        for c_idx in xrange(pred.shape[1]):
            if pred[n_idx,c_idx] > tmp_max:
                tmp_max = pred[n_idx,c_idx]
                tmp_idx = c_idx
        y_list.append(tmp_idx)
    return y_list

def checkout(pred,y):
    N = len(y)
    acc = 0
    for idx in xrange(N):
        if pred[idx] == y[idx]:
            acc += 1
    print acc*1. / N

if __name__ == "__main__":
    word_dict = get_voc_set()
    tr_x,tr_y = get_voc_vector(word_dict,tr_data_path,1,True)
    te_x,te_y = get_voc_vector(word_dict,te_data_path,1,True)
    
    #logreg = linear_model.LogisticRegression()
    logreg = LogisticRegression(0.5,0,0)
    logreg.fit(tr_x,tr_y)
    pred_y = logreg.predict(te_x)
    p_y = get_pred(pred_y)
    y = get_pred(te_y)
    checkout(p_y,y)

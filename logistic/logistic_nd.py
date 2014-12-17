#coding=utf-8

import pickle
import numpy as np
from read_conf import config
from math import exp,log

data_path = config('../conf/dp.conf')

class LogisticRegression(object):
    def __init__(self,rate=0.5,l1=0,l2=0):
        self.rate = rate
        self.l1 = l1
        self.l2 = l2

    def fit(self,x,y):
        self.D = x.shape[1]
        self.C = y.shape[1]
        if self.C == 1:
            self.g = np.matrix([0.] * self.D).reshape(self.D,1)
            self.w = np.matrix([0.] * self.D).reshape(self.D,1)
        elif self.C >= 3:
            self.g = np.matrix([0.] * self.D * self.C).reshape(self.D,self.C)
            self.w = np.matrix([0.] * self.D * self.C).reshape(self.D,self.C)
        else:
            print "Wrong CATE Demension.."
            exit(0)
        w = self.w
        g = self.g
        #一个循环，loss还是啥，预测，loss，更新...
        p = self.predict(x)
        p_loss = log_loss(p,y)
        d_loss = p_loss
        self.update(x,y,p)
        times = 10
        while d_loss >= 0.00005 and times > 0:
            p = self.predict(x)
            n_loss = log_loss(p,y)
            self.update(x,y,p)
            d_loss = n_loss - p_loss
            p_loss = n_loss
            print times
            times -= 1

    def predict(self,x):
        w = self.w
        wx = x * w
        if self.C == 1:
            for i in xrange(x.shape[0]):
                wx[i,0] = 1./(1.+exp(-max(min(wx[i,0],35.),-35.)))
        else:
            for i in xrange(wx.shape[0]):
                for j in xrange(wx.shape[1]):
                    #wx[i,j] = 1./(1.+exp(-wx[i,j]))
                    wx[i,j] = 1. / (1. + exp(-max(min(wx[i,j],35.),-35)))
        return wx

    def update(self,x,y,p):
        w = self.w
        g = self.g
        #p = self.predict(x)
        g = x.transpose() * (p - y)
        w -= self.rate * g

def log_loss(p,y):
    logloss = 0.
    if y.shape[1] == 1:
        for n_idx in xrange(y.shape[0]):
            logloss = -log(p[n_idx,0]) if y[n_idx,0] == 1. else -log(1.-p[n_idx,0])
    else:
        for n_idx in xrange(y.shape[0]):
            for c_idx in xrange(y.shape[1]):
                logloss += -log(p[n_idx,c_idx]) if y[n_idx,c_idx] == 1. else -log(1.-p[n_idx,c_idx])
    return logloss

def get_matrix(path):
    infile = open(path,'rb')
    tmp_matrix = pickle.load(infile)
    return tmp_matrix

def rebuild(tr_cat):
    if set(tr_cat) == 2:
        return np.matrix(np.array(tr_cat).reshape(len(tr_cat),1))
    else:
        c = len(set(tr_cat))
        tmp = np.array([0.]*len(tr_cat)*c).reshape(len(tr_cat),c)
        for idx in xrange(len(tr_cat)):
            tmp[idx][tr_cat[idx]] = 1.
        return np.matrix(tmp)

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

def get_acc(pred,te):
    acc_c = 0
    for idx in xrange(len(te)):
        if pred[idx] == te[idx]:
            acc_c += 1
    print acc_c,len(te)
    print "acc : %0.2f"%(1.*acc_c/len(te))
            

if __name__ == "__main__":
    tr_matrix = np.matrix(get_matrix(data_path['train_matrix']).toarray())
    tr_cat = rebuild(get_matrix(data_path['train_cat']))
    logreg = LogisticRegression(0.5,0,0)
    logreg.fit(tr_matrix,tr_cat)
    te_matrix = np.matrix(get_matrix(data_path['test_matrix']).toarray())
    te_cat = get_matrix(data_path['test_cat'])
    pred_y = logreg.predict(te_matrix)
    pred_list = get_pred(pred_y)
    get_acc(pred_list,te_cat)

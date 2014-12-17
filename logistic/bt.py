#coding=utf-8

import pickle
import numpy as np
from read_conf import config
from math import exp,log

data_conf = config('../conf/dp.conf')

class logistic(object):
    def __init__(self,L1,L2,lambda1,D):
        self.L1 = L1
        self.L2 = L2
        self.lambda1 = lambda1
        self.D = D

        self.g = np.array([0.] * D)
        self.w = np.array([0.] * D)
    
    def _indices(self,x):
        for idx in x:
            yield idx

    def predict(self,x):
        w = self.w
        pred_list = []
        wTx = 0.
        for x_line in self._indices(x):
            for idx in xrange(self.D):
                wTx += w[idx] * x_line[idx]
            pred_list.append(1. / (1. + exp(-max(min(wTx,35.),-35.))))
        return np.array(pred_list)

    def update(self,x,y,p):
        w = self.w
        g = self.g

        #p = self.predict(x)
        for w_idx in xrange(self.D):
            g_tmp = 0.
            for y_idx in xrange(len(y)):
                g_tmp += (p[y_idx] - y[y_idx]) * x[y_idx][w_idx]
            g[w_idx] = self.lambda1 * g_tmp
        print w
        w -= g
        print w

    def fit(self,x,y):
        w = self.w
        g = self.g
        for times in xrange(5):
            print " %s time"%times
            loss = 0.
            p = self.predict(x)
            print log_loss(p,y)
            self.update(x,y,p)

def log_loss(p,y):
    logloss = 0.
    for idx in xrange(len(y)):
        tmp = -log(p[idx]) if y[idx] == 1. else -log(1. - p[idx])
        logloss += tmp
    return logloss

def get_matrix(path):
    infile = open(path,'rb')
    tmp_matrix = pickle.load(infile)
    return tmp_matrix

if __name__ == "__main__":
    tr_matrix = get_matrix(data_conf['train_matrix']).toarray()
    tr_cat = np.array(get_matrix(data_conf['train_cat']))
    D = len(tr_matrix[0])
    log_reg = logistic(0,0,0.5,D)
    log_reg.fit(tr_matrix,tr_cat)

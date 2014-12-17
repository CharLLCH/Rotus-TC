#coding=utf-8
from word import word
import csv

w = word('cc',0,{'ex':1})

w1 = word('cj',0,{'e':1})

w.update_dict('ex')

w1.update_dict('ee')

w1.update_dict('ex')

w2 = word('cs',0)

w2.update_dict('ee')

print w1.get_wcatdict()
print w.get_wcatdict()
print w2.get_wcatdict()

with open("../../result/word_set.csv",'rb') as infile:
    reader = csv.reader(infile)


writer = csv.writer(open("../../result/test.csv",'wb'),quoting=csv.QUOTE_ALL)

writer.writerow(['12','21'])
writer.writerow([["12","21"]])

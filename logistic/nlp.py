#coding=utf-8
'''
我也来封装一下呗
'''

from nltk import regexp_tokenize
from nltk.stem import WordNetLemmatizer
#import textblob
#from textblob.tokenizers import SentenceTokenizer as sent_tok
#from textblob.tokenizers import WordTokenizer as word_tok
from read_conf import config

stopwords = open(config('../conf/dp.conf')['stopword_path'])
stopwords = stopwords.readlines()
stopwords = [item.strip() for item in stopwords]

pattern = r'''[a-zA-Z]+'''

class NLP(object):
    def __init__(self):
        #self.__wordnetlem = WordNetLemmatizer()
        #self.__stokenizer = sent_tok()
        #self.__wtokenizer = word_tok()
        self.__stopwords = set(stopwords)

    def word_tokenize(self,document):
        tokens = regexp_tokenize(document,pattern)
        tokens = [item.lower() for item in tokens]
        tokens = [item for item in tokens if item not in stopwords]
        return tokens
'''
    def sentences_tokenize(self,sents):
        sentences = self.__stokenizer.tokenize(sents)
        return sentences
    
    def words_tokenize(self,sent):
        return self.__wtokenizer.tokenize(sent)
'''

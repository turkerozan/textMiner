# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 23:56:03 2018

@author: Dev
"""
from __future__ import print_function
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer 
import re
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import gensim 
import numpy as np
from gensim.models import KeyedVectors
stemmer = SnowballStemmer('english')
words = stopwords.words("english")
#docs = ['yol yapdı', 'aman tanrım o yea', 'free style mı kanka bu','hehe lahmacun yiyah','kılışdar yol yol yol Yapdı','bu bu bu']
newlst= []
i=0
docs = pd.read_csv("kucuk.csv")

for index, row in docs.iterrows():
    
    columns = row.iloc[i].split("\t")
    #txt = columns[0]
    txt =  columns[1] + ' ' + columns[-1]
    txt = re.sub(r'[^\w\s]','',txt)
    newlst.append(txt)
    

word2vec_model = KeyedVectors.load_word2vec_format('wiki.vec')

def avg_feature_vector(words, model, num_features, index2word_set):
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    index2word_set = set(model.wv.index2word)# this is moved as input param for performance reasons

    for word in words:
        if word in index2word_set:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])
            print(word)
            #print(featureVec)
            
    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    featureVec = np.reshape(featureVec,(300,1))
    featureVec = np.transpose(featureVec)
    return featureVec
i = 0;

for doc in newlst:
    if(i == 0):
        featureVec = avg_feature_vector(doc.split(),model = word2vec_model, num_features=300, index2word_set = set(word2vec_model.wv.index2word))
        i = i + 1
    else:
        featureVec = np.append(featureVec, avg_feature_vector(doc.split(),model = word2vec_model, num_features=300, index2word_set = set(word2vec_model.wv.index2word)),axis=0)

df = pd.DataFrame(featureVec)
df.to_csv("fasttext.csv", index=False)

# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 21:57:34 2018

@author: Dev
"""

from __future__ import print_function
import gensim 
import numpy as np
from gensim.models import KeyedVectors
w2v_model = KeyedVectors.load_word2vec_format('wiki.vec')



def avg_features(words, model, num_features, i2w_s):
    #function to average all words vectors in a given paragraph
    featureVec = np.zeros((num_features,), dtype="float32")
    nwords = 0
    i2w_s = set(model.wv.i2w)# this is moved as input param for performance reasons

    for word in words:
        if word in i2w_s:
            nwords = nwords+1
            featureVec = np.add(featureVec, model[word])
            print(word)
            print(featureVec)
            
    if nwords>0:
        featureVec = np.divide(featureVec, nwords)
    return featureVec

sentence_1 = "this is sentence"
sentence_1_avg_vector = avg_features(sentence_1.split(), model=w2v_model, num_features=300,i2w_s=set(w2v_model.wv.i2w))
print(sentence_1_avg_vector)
print("---------\n")
#get average vector for sentence 2  
sentence_2 = "this is sentence sajkdhasjh3"
sentence_2_avg_vector = avg_features(sentence_2.split(), model=w2v_model, num_features=300,i2w_s=set(w2v_model.wv.i2w))
print(sentence_2_avg_vector)

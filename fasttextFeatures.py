# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 21:57:34 2018

@author: Dev
"""

from __future__ import print_function
import gensim 
import numpy as np
from gensim.models import KeyedVectors
#model = gensim.models.Doc2Vec.load('saved_doc2vec_model')  
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
    featureVec =np.transpose(featureVec)
    return featureVec

sentence_1 = "this is sentence"
sentence_1_avg_vector = avg_feature_vector(sentence_1.split(), model=word2vec_model, num_features=300,index2word_set=set(word2vec_model.wv.index2word))
print(sentence_1_avg_vector)
print("---------\n")
#get average vector for sentence 2  
sentence_2 = "this is sentence sajkdhasjh3"
sentence_2_avg_vector = avg_feature_vector(sentence_2.split(), model=word2vec_model, num_features=300,index2word_set=set(word2vec_model.wv.index2word))
print(sentence_2_avg_vector)
df = pd.DataFrame(sentence_2_avg_vector)
df.to_csv("fasttext.csv", index=False)

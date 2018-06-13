# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 00:01:58 2018

@author: Dev
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

docs = ['yol yapdı', 'aman tanrım o yea', 'free style mı kanka bu','hehe lahmacun yiyah','kılışdar yol yol yol Yapdı']
ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(5, 5))
ngram_vectorizer.fit_transform(docs)
#print(ngram_vectorizer.get_feature_names())


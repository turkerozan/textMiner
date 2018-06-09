# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 22:30:39 2018

@author: Dev_Ozan
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

docs = ['yol yapdı', 'aman tanrım o yea', 'free style mı kanka bu','hehe lahmacun yiyah','kılışdar yol yol yol Yapdı']
vec = CountVectorizer()
X = vec.fit_transform(docs)
binaryX = X.toarray()

for i in range(len(binaryX)):
    for j in range(len(binaryX[i])):
        if(binaryX[i][j] != 0):
            binaryX[i][j] = 1

df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
df2 = pd.DataFrame(binaryX, columns=vec.get_feature_names())

print(df)
print(df2)

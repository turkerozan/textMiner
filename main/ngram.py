# -*- coding: utf-8 -*-
"""
Created on Sat Jun  9 23:15:25 2018

@author: Dev
"""

from nltk import word_tokenize, ngrams
s = "foo bar sentence"
gg = list(ngrams(s, 2))


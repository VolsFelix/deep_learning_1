# project1

from codecs import lookup_error
from unicodedata import category
import numpy as np
import os
from matplotlib import pyplot as plt
import gzip
import numpy as np
import pandas as pd
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import nltk
#nltk.download('punkt')
from collections import Counter

#os.chdir("C:\\Users\\cole\\Documents\\Spring MSBA")

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

unique_titles= []
title_list=[]
category_list=[]

i = 0
titles_batch_list=[]
category_batch_list=[]
for d in parse('meta_Clothing_Shoes_and_Jewelry.json.gz'):
    i += 1
    X = (d['title'])
    title_list.append(X)
    titles_batch_list.append(X)
    Y = (d['category'])
    category_list.append(Y)
    category_batch_list.append(Y)
    if i == 5000:
        break

#tokenizing the words of each title and removing punctuation
tokenizer = nltk.RegexpTokenizer(r"\w+")

tokenized_titles_batch = [tokenizer.tokenize(i) for i in titles_batch_list]
#tokenized_title_list=[tokenizer.tokenize(i) for i in title_list]
len(titles_batch_list)
len(title_list)
len(unique_titles)
#getting rid of duplicate words in titles
T_cat_flat = [item for subcat in tokenized_titles_batch for item in subcat]
unique_titles= np.unique(np.array(T_cat_flat))

#############################

y_cat_flat = [item for subcat in category_list for item in subcat]
unique_categories = np.unique(np.array(y_cat_flat))
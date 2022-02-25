# project1

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


titles=[]
categories=[]
i = 0


for d in parse('meta_Clothing_Shoes_and_Jewelry.json.gz'):
    i += 1
    X = (d['title'])
    titles.append(X)
    Y = np.array(d['category'])
    categories.append(Y)
    if i == 100:
        break

#tokenizing the words of each title and removing punctuation
tokenizer = nltk.RegexpTokenizer(r"\w+")

tokenized_sents = [tokenizer.tokenize(i) for i in titles]
#getting rid of duplicate titles
unique_titles= []

for sentence in tokenized_sents:
    for word in sentence:
        if word not in unique_titles:
            unique_titles.append(word)


vectorizer = CountVectorizer(min_df=0, lowercase=False)
print(vectorizer.get_feature_names)
vectorizer.fit(unique_titles)
vectorizer.vocabulary_
X=vectorizer.transform(titles)
Hotcoded_X = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())
Hotcoded_X.head()

#checking to make sure no duplicate columns. It checks out
df = Hotcoded_X.loc[:,~Hotcoded_X.columns.duplicated()]
df.head()

#checking that rows 1 in the columns add up to the amount of words in the title
Hotcoded_X.iloc[99].sum()
titles[99]

 
### Office hours
## feed in X variables 1 by 1. single row is 1X50000. Make list of unique cats, 10000. matrix is nx10000. set value to 1 where corresponding cat
#50000 x 100000 y every x is 1 except for where the column corresponds to the word in the title. 
# for category not looking at words, looking at entire category

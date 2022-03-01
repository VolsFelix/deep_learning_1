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
    Y = (d['category'])
    categories.append(Y)
    if i == 100:
        break

#tokenizing the words of each title and removing punctuation
tokenizer = nltk.RegexpTokenizer(r"\w+")

tokenized_sents = [tokenizer.tokenize(i) for i in titles]



#getting rid of duplicate words in titles
unique_titles= []
for sentence in tokenized_sents:
    for word in sentence:
        if word not in unique_titles:
            unique_titles.append(word)
#############################

cat_list=[]
unique_cats=[]
##### Making list of categories
for i in categories:
    for j in i:
        cat_list.append(j)

### Function to find the max length of the categories for padding
def lengths(x):
    if isinstance(x,list):
        yield len(x)
        for y in x:
            yield from lengths(y)
max(lengths(cat_list))
#####
##### counting and plotting the most common categories
from collections import Counter
C=(Counter(cat_list))



#Padding Cat list 
for i in range(len(cat_list)):
    cat_list[i]=cat_list[i].ljust(max(lengths(cat_list)), '0')

#PADDING CATEGORIES
for i in range(len(categories)):
    for j in range(len(categories[i])):
        categories[i][j]=categories[i][j].ljust(max(lengths(cat_list)), '0')

### padded unique cats       
for b in cat_list:
    if b not in unique_cats:
        unique_cats.append(b)


vectorizer = CountVectorizer(min_df=0, lowercase=False)
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


##make each entry of categories into a string for Y
char_cat=[]
for i in range(len(categories)):
    char_cat.append(str(categories[i]))

 ########### category vecorization
vectorizer = CountVectorizer(min_df=0, lowercase=False,ngram_range=(max(lengths(cat_list)),max(lengths(cat_list))), analyzer='char')
vectorizer.fit(unique_cats)
vectorizer.vocabulary_
Y=vectorizer.transform(char_cat)
###
#### hot coding Y



Hotcoded_Y = pd.DataFrame(Y.toarray(),columns=vectorizer.get_feature_names())
Hotcoded_Y.head()
Hotcoded_Y.iloc[0].sum()
len(categories[0])
len(Hotcoded_Y)
categories[0]

### Office hours
## feed in X variables 1 by 1. single row is 1X50000. Make list of unique cats, 10000. matrix is nx10000. set value to 1 where corresponding cat
#50000 x 100000 y every x is 1 except for where the column corresponds to the word in the title. 
# for category not looking at words, looking at entire category

Hotcoded_X.head()
Hotcoded_Y.head()

indices = np.array(range(len(unique_cats)), dtype = np.int64)
indices
lookuptable = np.column_stack([unique_cats,indices])


inputs = tf.keras.layers.Input(shape=(Hotcoded_X.shape[1],), name='input') #Note: shape is a tuple and does not includes records. For a two dimensional input dataset, use (Nbrvariables,). We would use the position after the comma, if it would be a 3-dimensional tensor (e.g., images). Note that (something,) does not create a second dimension. It is just Python's way of generating a tuple (which is required by the Input layer).
hidden1 = tf.keras.layers.Dense(units=555, activation="relu", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=555, activation="relu", name= 'hidden2')(hidden1)
outputs = tf.keras.layers.Dense(units=356, activation = "sigmoid", name= 'output')(hidden2)

model = tf.keras.Model(inputs = inputs, outputs = outputs)
model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))
model.fit(x=Hotcoded_X,y=Hotcoded_Y, batch_size=1, epochs=50)
yhat = model.predict(x=Hotcoded_X)
model.evaluate(x=Hotcoded_X,y=Hotcoded_Y)

yhat[0].sum()
sorted(range(len(yhat[50])), key=lambda i: yhat[0][i])[-5:]

categories[50]
titles[50]
cat_list[148]
lookuptable[148]
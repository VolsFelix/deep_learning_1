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


len(category_list)
len(title_list[0])

# change this when adding more batches
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
    if i == 1000:
        break

#tokenizing the words of each title and removing punctuation
tokenizer = nltk.RegexpTokenizer(r"\w+")

tokenized_titles_batch = [tokenizer.tokenize(i) for i in titles_batch_list]
#tokenized_title_list=[tokenizer.tokenize(i) for i in title_list]
len(titles_batch_list)
len(title_list)
len(unique_titles)
#getting rid of duplicate words in titles

for sents in tokenized_titles_batch:
    for word in sents:
        if word not in unique_titles:
            unique_titles.append(word)
#############################

# cat_list=[]

##### Making list of categories
# for i in categories:
#     for j in i:
#         cat_list.append(j)

### Function to find the max length of the categories for padding
# def lengths(x):
#     if isinstance(x,list):
#         yield len(x)
#         for y in x:
#             yield from lengths(y)
# max(lengths(cat_list))
#####
##### counting and plotting the most common categories
from collections import Counter
for i in category_list:
    Counter(i)
category_list[0]
cleaned = [item for item in category_list if not isinstance(item,list)]
# len(max(cat_list))


# #Padding Cat list 
# for i in range(len(category_list)):
#     category_list[i]=category_list[i].ljust(len(max(category_list)), '0')

#PADDING CATEGORIES
# for i in range(len(categories)):
#     for j in range(len(categories[i])):
#         categories[i][j]=categories[i][j].ljust(max(lengths(cat_list)), '0')

### padded unique cats 
# 
# unique_cats=[]      
# for b in category_batch_list:
#     for i in b:
#         if i not in unique_cats:
#             unique_cats.append(b)




vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(unique_titles)
vectorizer.vocabulary_
X=vectorizer.transform([title_list[0]])
Hotcoded_X = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())
Hotcoded_X.head()

#checking to make sure no duplicate columns. It checks out
df = Hotcoded_X.loc[:,~Hotcoded_X.columns.duplicated()]
df.head()

#checking that rows 1 in the columns add up to the amount of words in the title
Hotcoded_X.iloc[0].sum()
title_list[0]


##make each entry of categories into a string for Y
# char_cat=[]
# for i in range(len(categories)):
#     char_cat.append(str(categories[i]))

 ########### category vecorization
# vectorizer = CountVectorizer(min_df=0, lowercase=False,ngram_range=(len(max(unique_cats)),len(max(unique_cats))), analyzer='char')
# vectorizer.fit(unique_cats)
# vectorizer.vocabulary_
# Y=vectorizer.transform(char_cat)
###
#### hot coding Y



# Hotcoded_Y = pd.DataFrame(Y.toarray(),columns=vectorizer.get_feature_names())
# Hotcoded_Y.head()
# Hotcoded_Y.iloc[0].sum()
# len(categories[0])
# len(Hotcoded_Y)
# categories[0]

### Office hours
## feed in X variables 1 by 1. single row is 1X50000. Make list of unique cats, 10000. matrix is nx10000. set value to 1 where corresponding cat
#50000 x 100000 y every x is 1 except for where the column corresponds to the word in the title. 
# for category not looking at words, looking at entire category

y_cat_flat = [item for subcat in category_list for item in subcat]
unique_categories = np.unique(np.array(y_cat_flat))
len(unique_categories)
indices = np.array(range(len(unique_categories)), dtype = np.int64)
indices
lookuptable = np.column_stack([unique_categories,indices])
lookuptable


#step 3: apply lookup table to data
main_result = []

res = []
for ii in range(len(category_list[0])):
    res.append(int(lookuptable[lookuptable[:,0]==category_list[0][ii],1][0]))
main_result.append(res)

main_result
lookuptable[376]

#### added 
main_result = []
for i in range(len(category_list)):
    res = []
    for ii in range(len(category_list[i])):
        res.append(int(lookuptable[lookuptable[:,0]==category_list[i][ii],1][0]))
    main_result.append(res)
main_result


#step 4: create dummy encoded data

y_final = np.array([list(np.zeros(len(unique_categories))) for i in range(len(category_list[0]))])
for i in range(len(category_list[0])):
    for ii in range(len(main_result[i])):
        y_final[i,main_result[i][ii]] = 1 


cheese=pd.DataFrame(y_final).T
y_final_individual=y_final[0]

ham= pd.DataFrame([y_final[0]])


inputs = tf.keras.layers.Input(shape=Hotcoded_X.shape[1], name='input') #Note: shape is a tuple and does not includes records. For a two dimensional input dataset, use (Nbrvariables,). We would use the position after the comma, if it would be a 3-dimensional tensor (e.g., images). Note that (something,) does not create a second dimension. It is just Python's way of generating a tuple (which is required by the Input layer).
hidden1 = tf.keras.layers.Dense(units=Hotcoded_X.shape[1], activation="sigmoid", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=Hotcoded_X.shape[1], activation="relu", name= 'hidden2')(hidden1)
outputs = tf.keras.layers.Dense(units=y_final.shape[1], activation = "sigmoid", name= 'output')(hidden2)

model = tf.keras.Model(inputs = inputs, outputs = outputs)
model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))
model.fit(x=Hotcoded_X,y=ham, batch_size=1, epochs=50)
yhat = model.predict(x=Hotcoded_X)
model.evaluate(x=Hotcoded_X,y=ham)

title_list[0]
yhat
sorted(range(len(yhat[0])), key=lambda i: yhat[0][i])[-5:]
category_batch_list[0]
lookuptable[1070]


#######################################
#Create some nonlinear toy data.
import matplotlib.pyplot as plt
import numpy as np
ct = np.ones(20) 
X1 = np.random.normal(size=20) #variable, 20 rows
X2 = np.random.normal(size=20) #variable, 20 rows
X = np.array(np.column_stack((X1,X2)))
y = ct*2.2222 + X1*5.4675 + X2*10.1115 - 3*X1**2

ycat = []
for i in y:
    if i <= np.min(y):
        ycat.append(['cat 0'])
    elif i <= np.quantile(y,0.25):
        ycat.append(['cat 1','cat 2'])
    elif i <= np.quantile(y,0.50):
        ycat.append(['cat 2'])        
    elif i <= np.quantile(y,0.75):        
        ycat.append(['cat 3'])        
    elif i <= np.max(y):        
        ycat.append(['cat 4'])

ycat
y_cat_flat = [item for subcat in ycat for item in subcat]
unique_categories = np.unique(np.array(y_cat_flat))
unique_categories

##### changed unique_cats to indexs
indices = np.array(range(len(unique_cats)), dtype = np.int64)
indices
lookuptable = np.column_stack([unique_cats,indices])
lookuptable[0]
categories
#step 3: apply lookup table to data
main_result = []
for i in range(len(categories)):
    res = []
    for ii in range(len(categories[i])):
        res.append(int(lookuptable[lookuptable[:,0]==categories[i][ii],1][0]))
    main_result.append(res)

main_result

#step 4: create dummy encoded data

y_final = np.array([list(np.zeros(len(unique_cats))) for i in range(len(categories))])
for i in range(len(categories)):
    for ii in range(len(main_result[i])):
        y_final[i,main_result[i][ii]] = 1        

y_final
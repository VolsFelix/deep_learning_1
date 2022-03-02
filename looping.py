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
from dlproject1 import model,unique_titles,title_list,category_list,category_batch_list,lookuptable,unique_categories,yh


for j in range(20):
    
    #set up x hotcode
    vectorizer = CountVectorizer(min_df=0, lowercase=False)
    vectorizer.fit(unique_titles)
    vectorizer.vocabulary_
    X=vectorizer.transform([title_list[0]])
    Hotcoded_X = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())
    Hotcoded_X.head()

    #set up y hotcode
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


    #cheese=pd.DataFrame(y_final).T
    y_final_individual=y_final[j]

    ham= pd.DataFrame([y_final[j]])



    #model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))
    model.fit(x=Hotcoded_X,y=ham, batch_size=1, epochs=10)
    yhat = model.predict(x=Hotcoded_X)
    model.evaluate(x=Hotcoded_X,y=ham)
    model.evaluate(x=Hotcoded_X,y=ham)
    yhat = model.predict(x=Hotcoded_X)
    yh.append(yhat)



yh

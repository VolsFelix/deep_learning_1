# project1

import numpy as np
import gzip
import os
import tensorflow as tf
from tensorflow import keras
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import TextVectorization
import re
import string
import random
from sklearn.model_selection import train_test_split

os.chdir("C:\\Users\\cole\\Documents\\Spring MSBA")
os.chdir("C:\\Users\\kevin\\OneDrive\\Documents\\BZAN 554 Deep Learning")

def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

i = 0
df = {}
for d in parse('meta_Clothing_Shoes_and_Jewelry.json.gz'):
    i += 1
    X = np.array(d['title'])
    print('X (title):\n')
    print(X)
    Y = np.array(d['category'])
    print('\nY (category):\n')
    print(Y)
    if i == 10:
     break


tr_stop_words = pd.read_json('meta_Clothing_Shoes_and_Jewelry.json.gz')
for each in tr_stop_words.values[:5]:
  print(each[0])

#Step 5 LOAD THE DATASET
data = pd.read_json('meta_Clothing_Shoes_and_Jewelry.json.gz')


#Step 6  EXPLORE THE DATASET
 #tf.keras.layers.TextVectorization
 
data= data.sample(frac=1)


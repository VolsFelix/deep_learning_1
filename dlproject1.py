# project1

import numpy as np
import gzip
import os
#os.chdir("C:\\Users\\cole\\Documents\\Spring MSBA")
#os.chdir("C:\\Users\\kevin\\OneDrive\\Documents\\BZAN 554 Deep Learning")

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

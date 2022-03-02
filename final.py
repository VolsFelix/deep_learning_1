from matplotlib.pyplot import hot
from import_unique import unique_categories, unique_titles, indices, lookuptable,title_list,category_list

i = 0
title_list=[]
category_list=[]
for d in parse('meta_Clothing_Shoes_and_Jewelry.json.gz'):
    i += 1
    X = (d['title'])
    title_list.append(X)
    Y = (d['category'])
    category_list.append(Y)
    if i == 100:
        break

for i in range(len(category_list)):
    category_list[i]=([x for x in category_list[i] if x in unique_categories])
len(category_list[1])


######################3 RUN ONCE
vectorizer = CountVectorizer(min_df=0, lowercase=False)
vectorizer.fit(unique_titles)
vectorizer.vocabulary_

#####################################
X=vectorizer.transform(title_list)
Hotcoded_X = pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names())
Hotcoded_X.head()
len(Hotcoded_X)
Hotcoded_X.iloc[0].sum()
len(title_list)


main_result = []
for i in range(len(category_list)):
    res = []
    for ii in range(len(category_list[i])):
        res.append(int(lookuptable[lookuptable[:,0]==category_list[i][ii],1][0]))
    main_result.append(res)
main_result


y_final = np.array([list(np.zeros(len(unique_categories))) for i in range(len(category_list))])
for i in range(len(category_list)):
    for ii in range(len(main_result[i])):
        y_final[i,main_result[i][ii]] = 1 




inputs = tf.keras.layers.Input(shape=Hotcoded_X.shape[1], name='input') 
hidden1 = tf.keras.layers.Dense(units=Hotcoded_X.shape[1], activation="elu", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=Hotcoded_X.shape[1], activation="elu", name= 'hidden2')(hidden1)
outputs = tf.keras.layers.Dense(units=y_final.shape[1], activation = "elu", name= 'output')(hidden2)


model = tf.keras.Model(inputs = inputs, outputs = outputs)
model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))
model.fit(x=Hotcoded_X,y=y_final, batch_size=1, epochs=1)
yhat = model.predict(x=Hotcoded_X)
model.evaluate(x=Hotcoded_X,y=y_final)

sorted(range(len(yhat[0])), key=lambda i: yhat[0][i])[-5:]



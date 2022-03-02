from matplotlib.pyplot import hot
from import_unique import unique_categories, unique_titles, indices, lookuptable,title_list,category_list
from sklearn.model_selection import train_test_split
import eli5
from eli5.sklearn import PermutationImportance


i = 0
title_list=[]
category_list=[]
for d in parse('meta_Clothing_Shoes_and_Jewelry.json.gz'):
    i += 1
    X = (d['title'])
    title_list.append(X)
    Y = (d['category'])
    category_list.append(Y)
    if i == 500:
        break

for i in range(len(category_list)):
    category_list[i]=([x for x in category_list[i] if x in unique_categories])
len(category_list[1])


######################3 RUN ONCE TO CREAT HOTCODED COLUMNS
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



X_train,X_test, Y_train, Y_test=train_test_split(Hotcoded_X,y_final, test_size=0.2)

inputs = tf.keras.layers.Input(shape=Hotcoded_X.shape[1], name='input') 
hidden1 = tf.keras.layers.Dense(units=Hotcoded_X.shape[1], activation="elu", name = 'hidden1')(inputs)
hidden2 = tf.keras.layers.Dense(units=Hotcoded_X.shape[1], activation="elu", name= 'hidden2')(hidden1)
outputs = tf.keras.layers.Dense(units=y_final.shape[1], activation = "elu", name= 'output')(hidden2)


model = tf.keras.Model(inputs = inputs, outputs = outputs)
model.compile(loss = 'binary_crossentropy', optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001))
model.fit(x=X_train,y=Y_train, batch_size=1, epochs=1)
yhat = model.predict(x=X_test)
model.evaluate(x=X_test,y=Y_test)

sorted(range(len(yhat[0])), key=lambda i: yhat[0][i])[-5:]


lookuptable[]
Hotcoded_X


x = X_test
y = y_final
#VARIABLE IMPORTANCE ON TRAINING DA
#step 2

performance_before = np.corrcoef(y,yhat)[0,1]
performance_before
#step 3
importance = list()
for ind in [1,2]:
    x_final_cp = np.copy(x)
    variable = np.random.permutation(np.copy(x.iloc[:,ind]))
    x_final_cp[:,ind] = variable
    yhat = model.predict(x_final_cp)
    performance_after = np.corrcoef(y,yhat)[0,1]
    importance.append(performance_before - performance_after)
importance
#DEPENDENCY PLOT
ind = 1
v = np.unique(x.iloc[:, ind])
means = []
for i in v:
    x_final_cp = np.copy(x)
    x_final_cp[:, ind] = 1
    yhat = model.predict(x)
    means.append(np.mean(yhat))
#PLOTTING PARTIAL DEPENDENCY
plt.plot(v,means)
plt.title("Dependency Plot")
plt.xlabel("Value of Variable" + str(ind))
plt.ylabel("Mean of Predicted Response")
plt.show()

perm = PermutationImportance(model,random_state=1).fit(X_test,Y_test)
model.show_weights()







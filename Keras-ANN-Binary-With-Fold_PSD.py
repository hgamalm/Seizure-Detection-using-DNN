import numpy as np
import os
y = np.zeros((200,))

mylist = []
f = open('PSD_Z_A.txt', 'r')
x1 = f.readlines()
mylist =x1
f = open('PSD_S_A.txt', 'r')
x2 = f.readlines()
mylist.extend(x2)
y[0:100] = 1    #F
y[100:200] = 0  #O
#y[200:300] = 0  #S

#y[300:400] = 0
#y[400:500] = 1

mylist = [float(i) for i in mylist]
X = np.array(mylist)
X= X.reshape(200,1)
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
# Feature Scaling
# Normalization
from sklearn.preprocessing import StandardScaler # for normalization
scaler = StandardScaler()
X= scaler.fit_transform(X)
# define 10-fold cross validation test harness
from keras.models import Sequential #Initialize ANN
from keras.layers import Dense  #Build Layers
from keras.layers import Dropout  # for use of Dropout for preventing overfitting
from sklearn.model_selection import StratifiedKFold
cmFinal = np.zeros((2,2))
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, y):
  # create model
  print(test)
  
  #internally dividing the training into train and validate
  X_train, X_test, y_train, y_test = train_test_split(X[train], y[train], test_size=0.2, random_state=seed)
  #print(y_test)
  print(108-np.count_nonzero(y_test))
  model = Sequential()
  # The below line means that I have 1 input as the input layer and 12 hidden units in the first layer which 
  #has the relu activation function
  model.add(Dense(12, input_dim=1, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(8, activation='relu'))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))
	# Compile model
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	# Fit the model
  model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=3)
  y_pred = model.predict(X[test])
  y_pred = (y_pred > 0.5)

  # Making the Confusion Matrix
  from sklearn.metrics import confusion_matrix
  cm = confusion_matrix(y[test], y_pred)
  cmFinal = cmFinal + cm
  print(cm)
  print(cmFinal)

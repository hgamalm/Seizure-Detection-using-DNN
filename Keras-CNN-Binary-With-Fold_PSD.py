import numpy as np
import os
y = np.zeros((200,))
mylist = []
f = open('PSD_O_A.txt', 'r')
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler # for normalization
scaler = StandardScaler()
X= scaler.fit_transform(X)
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# Splitting the dataset into the Training set and Test set
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.models import Sequential #Initialize ANN
from keras.layers import Dense  #Build Layers
from sklearn.model_selection import StratifiedKFold
from keras import optimizers
cmFinal = np.zeros((2,2))
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
cvscores = []
for train, test in kfold.split(X, y):
    print(test)
    X_train, x_validate, y_train, y_validate = train_test_split(X[train], y[train], test_size=0.3, random_state=seed)
    
    #Dimension Extension to be  3D Tensor
    X_train = np.expand_dims(X_train , axis=2)
    x_validate =  np.expand_dims(x_validate , axis=2)    
    # Initialising the CNN
    classifier = Sequential()
    
    # Layer 0-1
    classifier.add(Conv1D(4, kernel_size =1, input_shape=(1,1), activation = 'relu', strides=1  ))
    classifier.add(Flatten())
    classifier.add(Dense(units = 20, activation = 'relu'))
    classifier.add(Dense(units = 1, activation = 'sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Fit the model
    classifier.fit(X_train, y_train, validation_data=(x_validate,y_validate), epochs=150, batch_size=3)
    #Test per fold test data X[test]
    Y_Pred = classifier.predict(np.expand_dims(X[test] , axis=2))    
    #Convert predictions to 0 or 1
    Y_pred = (Y_Pred > 0.5)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y[test], Y_pred)
    cmFinal = cmFinal + cm
    print(cm)
    print(cmFinal)

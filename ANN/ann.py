# Artificial Neural Network
#The very step is to install these

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

script_dir = os.path.dirname("__file__")
print(script_dir)
abs_file_path = os.path.join(script_dir, 'Churn_Modelling.csv')
print(abs_file_path)

# Importing the dataset
dataset = pd.read_csv('C:\\Users\\ksred\\PycharmProjects\\Pyth1\\ANN\\Churn_Modelling.csv')
#X is independent values (starts from zero and upper bound is +1)
X = dataset.iloc[:, 3:13].values
#Y is dependent set (last coloumn) (no upperbound +1 is present here)
y = dataset.iloc[:, 13].values

# Encoding categorical data
#label encoder to convert those categorical data into digits
#onehotencoder is to skip the dummy variable trap (Only for coloumns containibg more than 2 values)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

#onehotencoding is being done here only for index=1 th coloumn, as it is the only one having more than 2 values.
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling (Just to bring every value into same scale)
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
# Sequential (the other one is graph) for initializing the ANN and dense to add the layers
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense
from tensorflow.contrib.keras import backend

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
# units = no nodes in the hidden layer (purely our wish. Normal :- (i/p + o/p)/2) Artist:- parameter tuning using k fold cross validation for exploration.
# kernel_initializer=uniform will initialize weights of nodes close to zero. There are also other fucntions such as glorot_uniform etc. For simplicity we will use uniform
#activation will the activation function we are gonna use for this particular layer
#input_dim will be the no. of inputs the layer is gonna get. We will do this only for first layer we will have to mention.
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))

# Adding the second hidden layer
classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
# This is the last layer, here we will choose sigmoid activation as it has a probabalistic ratio approach
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
# We have only initialized the weights, but we havent specified any algorithm to update weights.Here we since using stochastic gradient descent. There are many
# variants in it like that of Adagrad, Adadelta, RMSprop, Adam, AdaMax, Nadam,  AMSGrad. Adam is considered as efficient.
# loss is loss function to calculate the deviation of weights and based on the sigmoid func we are using in the ouput layer
# and based on the optimizer adam, here we are using the logarithmic loss function and name
# given to is binary_crossentropy( for o/ps = 2 or less). If more than two outcomes we will use categorical_crossentropy
# metrics refers to all the metric we are considering for the updation of weights. there can be many. We are considering only accuracy here.
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the Training set
#batch size= no of observations after which you wish to update the weights
# epochs= no. of vibrations
classifier.fit(X_train, y_train, batch_size=10, epochs=100, validation_split=0.1)

# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#this makes y_pred a boolean type one, which return true if y_pred is greteater than 0.5 and false if y_pred is less than 0.5
y_pred = (y_pred > 0.5)

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of Products: 2
Has Credit Card: Yes
Is Active Member: Yes
Estimated Salary: 50000"""
new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

#confusion matric which consists of diagonal of elements of right and wrong predictions

cm = confusion_matrix(y_test, y_pred)
print(cm)
backend.clear_session()


#Saving the weights
fname = "weights-ann.hdf5"
classifier.save_weights(fname, overwrite=True)

#loading the weights
fname = "weights-ann.hdf5"
classifier.load_weights(fname)
# Artificial Neural Network (Evaluated and Improved).
# Here the data pre-processing part will be same as the previous scenario.

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries (MNOP)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

script_dir = os.path.dirname('__file__')
abs_file_path = os.path.join(script_dir, 'Churn_Modelling.csv')

# Importing the dataset(these are pandas functions)
dataset = pd.read_csv('C:\\Users\\ksred\\PycharmProjects\\Pyth1\\ANN\\Churn_Modelling.csv')
#X is independent values (starts from zero and upper bound is +1)
X = dataset.iloc[:, 3:13].values
#Y is dependent set (last coloumn) (no upperbound +1 is present here)
y = dataset.iloc[:, 13].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#labelencoding to convert strings to digits
#onehotencoder is to skip the dummy variable trap (Only for coloumns containibg more than 2 values)

labelencoder_X_1 = LabelEncoder()

X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#onehotencoding to remove the dummy variable trap
#onehotencoding is being done here only for index=1 th coloumn, as it is the only one having more than 2 values.
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
#remove 1st coloumn(one dummy variable and it is useless. Theoritical idea= if not these two, it is the other one).
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set(text size=0.2 means, 20% of our records are used for testing and remaining 0.8 i.e 80% for training.

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling(to bring down all the values to one scale).
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# THE END OF PREPROCESSING.

# Evaluating, Improving and Tuning the ANN

# Evaluating the ANN
from sklearn.model_selection import cross_val_score #The actual k-fold cross validation function from scikics
from tensorflow.contrib.keras.api.keras.wrappers.scikit_learn import KerasClassifier #include k-fold classifer into keras(bcoz earlier the scenario is keras is from another stream and k-fold from scikic is of from another stream(so we neeed to integrate them both and wrap them into one).
from sklearn.model_selection import GridSearchCV #for parameter tuning of hyper parameters to increase the performance of our system.
from tensorflow.contrib.keras.api.keras.models import Sequential     #{These are the model initialization functions:- Dense ,Sequential and Dropout is to make sure that the machine doesnt overfit)
from tensorflow.contrib.keras.api.keras.layers import Dense, Dropout
from tensorflow.contrib.keras import backend

#sequential initiates the neurons and dense method helps add the input and hidden layers

#here we have to make use a function just becoz of the reason that the Keras Classifier we have imported above expects atleast one parameter to be the function.

#This function is meant to return the whole architecture of the classifier.
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

#Defining the classifier using wrapper i.e KerasClassifier.
classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=10)

#  Here we are actually applying the k-folds cross valid.
# cv = 10 is the usual number used for cross validation (it runs 10 different experiments)
# estimator= classifier we are gonna insert
# cv= number of folds ( 10-folds is always recommended)
# n_jobs = -1 (implies that all of the CPU's are used for computation) Since this is a very processing power demand process.
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=1) #cv means no. of folds and n_jobs=1 means use all cpus for the computation(to make it faster)

mean = accuracies.mean()
variance = accuracies.std()



# Tuning the ANN (deals with the parameter tuning for improving the  accuracy)
# For these we will be using dropout, which makes for every iteration some of the neurons will be automatically dropeed-out
# preventing the machine to overfit.
# It is simple all we need to add is a Dropout function after the layer you have populated inorder to drop some neurons in that layer each iteration (0.1 = 1 neuron will be i=dropped)(0.2,0.3...2 ,3 neurons and so on)(But dont go over 0.5).
def build_classifier(optimizer):
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim=11))
    # Improving the ANN
    # Dropout Regularization to reduce overfitting if needed
    classifier.add(Dropout(0.1))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

#This is the parameter tuning using hyper parameters, we are tuning various paramter values into the paramters varibles which can
# can be used as param_grid in the GridSearchCv function. It returns the values accuracy along with the best parameters it has found.
classifier = KerasClassifier(build_fn=build_classifier)
parameters = {'batch_size': [9],
              'epochs': [9],
              'optimizer': ['adam', 'rmsprop']}
grid_search = GridSearchCV(estimator=classifier,
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=12)
grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_


backend.clear_session()


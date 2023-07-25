'''
Purpose : Creating a fully connceted neural network, training it and selecting the best model 
Author  : Ayat Bhija 
'''

# IMPORTS 
import tensorflow as tf
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from Features_Extraction import extract_feature

# Datasets
path2load = '../numpy_dataset/'
_X_train = np.load(path2load + 'X_train.npy')
y_train = np.load(path2load + 'y_train.npy')
_X_val = np.load(path2load + 'X_val.npy')
y_val = np.load(path2load + 'y_val.npy')
_X_test = np.load(path2load + 'X_test.npy')
y_test = np.load(path2load + 'y_test.npy')

# Etracting Features
X_train = extract_feature(_X_train)
#X_train = X_train[:,:,:,np.newaxis]
X_val = extract_feature(_X_val)
X_test = extract_feature(_X_test)

# Creating a configurable neural network 
def create_model(neu):
  model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=(2, 16, 1)), 
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(neu, activation=tf.nn.relu),
  #tf.keras.layers.Dropout(0.1),
  tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)
  ])
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

# Variables
n_classes = len(y_train[0])
nb_features = len(X_train[1])
nb_feat = X_train.shape[1:]

# Defining the best model ---------------------------------------------------------------------- 
nb_neurones = [5, 10, 50, 100] # Defining a list of number of hidden neurons 
score_best = 0
model_best = None
for n in nb_neurones: # Testing for each number of neurons 
  model = create_model(n)
  #model.summary()
  model.fit(X_train, y_train, epochs=50, verbose=0) # Training 
  _, val_acc = model.evaluate(X_val, y_val, verbose=0) # Evaluation 
  print("Validation accuracy {} neurons: {}".format(n, val_acc))
  if val_acc > score_best:
      score_best = val_acc
      model_best = model

#------------------------------------------------------------------------ Saving the best model ---------------------------------------------------
#model_best.save('C:/Users/user0/Documents/ML_Algorithms/Cosmic dataset/FullyConnected/Models/model_FCNN.h5')
model_best.save('C:/Users/user0/Documents/ML_Aglorithms/Cosmic dataset/FullyConnected/Models/model_FCNN.h5')

# ----------------------------------------------------------------------- Displaying results ---------------------------------------------------------------------- 
model_best.summary()
model_best.evaluate(X_test, y_test, verbose=1)
print("")

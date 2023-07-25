#  This model contains 3 layers : input, hidden and output 
# The goal is to find the optimal number of neurons for the hidden layer 

#   // USE Model.py to create and train the model \\

#---------------------------------------------------------------- Imports -------------------------------------------------------------------------------------
import tensorflow as tf
from glob import glob
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler
from Features_Extraction import extract_feature
#--------------------------------------------------------------- Functions --------------------------------------------------------------------------------------
# Creating a configurable neural network 
def create_model(neu):
  model = tf.keras.models.Sequential([
  tf.keras.layers.Input(shape=((n_feat))), 
  # tf.keras.layers.Input(shape=((80,16))),
  # tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(neu, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.5),
  #tf.keras.layers.Dense(neu, activation=tf.nn.relu),
  tf.keras.layers.Dense(n_classes, activation=tf.nn.softmax)
  ])
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  return model

#------------------------------------------------------------- Loading the dataset ----------------------------------------------------------------------------
# Opening the dataset using glob
all_path = glob('./Dataset/Cubes/80_Samples/*/*')

# Creating 2 empty lists for the data and the labels 
X = [] #Data
y = [] #Labels

# Extracting the data and the labels --> Raw data 
for f in all_path:
    df = pd.read_csv(f, header=None)
    X.append(df._values)
    label = f.split('\\')[1]
    y.append(int(label))

# Transformimg the list to an array --> Easier to use
X = np.asarray(X)
y = np.asarray(y)

# Extracting the features from the data
X = extract_feature(X)

# Categorizing labels 
n_classes = len(np.unique(y))
y_cat = tf.keras.utils.to_categorical(y, num_classes=n_classes, dtype='float32')

# Normalizing the data of each sensor seperatly --> Better accuracy 
#X_resh = X.reshape(-1, X.shape[-1])
#scaler = MinMaxScaler()
#X_resh_norm = scaler.fit_transform(X_resh)
#X_norm = X_resh_norm.reshape(X.shape)

X_resh = X.reshape(X.shape[0],X.shape[1]*X.shape[2])
n_feat = X_resh.shape[-1]
scaler = MinMaxScaler()
X_norm = scaler.fit_transform(X_resh)

# Splitting the data : Train, Evaluate and Test data
X_train, X_tmp, y_train, y_tmp = train_test_split(X_norm, y_cat, stratify=y_cat, train_size=120/170, shuffle=True, random_state=666)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, stratify=y_tmp, test_size=0.5, shuffle=True, random_state=777)

# ---------------------------------------------------------- Defining the best model ---------------------------------------------------------------------- 
neu = [5, 10, 50, 100] # Defining a list of number of hidden neurons 
score_best = 0
model_best = None
for n in neu: # Testing for each number of neurons 
  model = create_model(n)
  #model.summary()
  model.fit(X_train, y_train, epochs=50, verbose=0) # Training 
  _, val_acc = model.evaluate(X_val, y_val, verbose=0) # Evaluation 
  print("Validation accuracy {} neurons: {}".format(n, val_acc))
  if val_acc > score_best:
      score_best = val_acc
      model_best = model

#------------------------------------------------------------------------ Saving the best model ---------------------------------------------------
model_best.save('C:/Users/user0/Documents/ML_Basics/Cosmic dataset/model_FCNN.h5')

# ----------------------------------------------------------------------- Displaying results ---------------------------------------------------------------------- 
model_best.summary()
model_best.evaluate(X_test, y_test, verbose=1)
print("")

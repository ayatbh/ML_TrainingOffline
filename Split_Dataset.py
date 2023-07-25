import numpy as np
from glob import glob
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler


# Opening the dataset using glob
all_path = glob('./Dataset/Cubes/80_Samples/*/*')
path2save = '.\\numpy_dataset\\'
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

# Categorizing labels 
n_classes = len(np.unique(y))
y_cat = tf.keras.utils.to_categorical(y, num_classes=n_classes, dtype='float32')

# Normalizing the data of each sensor seperatly --> Better accuracy 
X_resh = X.reshape(-1, X.shape[-1])
scaler = MinMaxScaler()
X_resh_norm = scaler.fit_transform(X_resh)
X_norm = X_resh_norm.reshape(X.shape)
X_norm = X_norm[:,:,:,np.newaxis]
# X_ = X_norm.reshape(X_norm.shape[0], X_norm.shape[1]*X_norm.shape[2])

# Splitting the data : Train, Evaluate and Test data
X_train, X_tmp, y_train, y_tmp = train_test_split(X_norm, y_cat, stratify=y_cat, train_size=120/170, shuffle=True, random_state=666)
X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, stratify=y_tmp, test_size=0.5, shuffle=True, random_state=777)

np.save(path2save + 'X_train.npy', X_train)
np.save(path2save + 'y_train.npy', y_train)
np.save(path2save + 'X_val.npy', X_val)
np.save(path2save + 'y_val.npy', y_val)
np.save(path2save + 'X_test.npy', X_test)
np.save(path2save + 'y_test.npy', y_test)

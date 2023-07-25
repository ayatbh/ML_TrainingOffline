'''
Author  : Ayat B.
Purpose : Extract features from a dataset  (or just a list of floats)
'''
import numpy as np 
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import MinMaxScaler

def extract_feature(X) : 
    # X is the data 
    Features = []
    for x in X : 
        tmp = []
        AVG = np.mean(x, axis=0)
        #NRJ = np.sum(x**2, axis=0)
        STD = np.std(x, axis=0)
        #MIN = np.min(x, axis=0)
        #MAX = np.max(x, axis=0)
        tmp.append(AVG)
        #tmp.append(NRJ)
        tmp.append(STD)
        #tmp.append(MIN)
        #tmp.append(MAX)
        Features.append(tmp)
    return np.asarray(Features)


# Testing the function 
# time_series_data = np.array( [1,2,3,4,5])
# extracted_features = extract_feature( time_series_data)

#print("Extracted Features:")
#for i, feature_value in enumerate(extracted_features):
#    feature_name = ['average', 'energy', 'std', 'min', 'max'][i]
#    print(f"{feature_name}: {feature_value}")

    
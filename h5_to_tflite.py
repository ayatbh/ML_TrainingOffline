'''
Author  : Ayat B.
Purpose : Converting h5 file to TFLite model 
'''
import tensorflow as tf
from tensorflow import keras
model = tf.keras.models.load_model('../Models/model_FCNN.h5')

converter = tf.lite.TFLiteConverter.from_keras_model(model)
#converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

open("../Models/model_FCNN.tflite", "wb").write(tflite_model)




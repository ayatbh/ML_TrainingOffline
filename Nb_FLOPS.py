'''
@purpose : The purpose of this file is to calculate the number of FLOPS of the ML model
@author  : Ayat Bhija
'''

import tensorflow as tf
import h5py

# Function to calculate FLOPs for a specific layer type
def calculate_flops_for_layer(layer):
    if isinstance(layer, tf.keras.layers.Conv2D):
        # FLOPs for 2D convolution layer
        flops = layer.kernel_size[0] * layer.kernel_size[1] * layer.filters * layer.input_shape[-1] * layer.output_shape[-1]
    elif isinstance(layer, tf.keras.layers.Dense):
        # FLOPs for fully connected layer (dense layer)
        flops = layer.input_shape[-1] * layer.units
    else:
        # Add more cases for other layer types if needed
        flops = 0
    return flops

# Function to calculate the total FLOPs for the model
def calculate_total_flops(model):
    total_flops = 0
    for layer in model.layers:
        layer_flops = calculate_flops_for_layer(layer)
        total_flops += layer_flops
    return total_flops

# Main code to load the model from h5 file and calculate FLOPs
if __name__ == "__main__":
    model = tf.keras.models.load_model('../Models/model_FCNN.h5')  # Load the h5 model file and reconstruct the model
    total_flops = calculate_total_flops(model)
    print("Total FLOPs:", total_flops)

import numpy as np
import tensorflow as tf

a = np.zeros((1, 80, 16, 1), dtype=np.float32)


def test_datum(x):
    if len(x.shape)==1:
        a[0, ] = x
    else:
        a[0, ] = x
    interpreter.set_tensor(input_details[0]['index'], a)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


interpreter = tf.lite.Interpreter(model_path='.//Models//CNN2D_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']



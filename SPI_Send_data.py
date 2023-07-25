'''
The purpose of this file is to send a data list  X_test to an STM32 Card
The ultimate goal is to test the CNN model on the nucleo card 

'''
#Imports
import os
os.environ["BLINKA_FT232H"] = "1"
import board
import digitalio
from pyftdi.spi import SpiController
import struct
import numpy as np
import time
import Run_tflite


X = np.load('.\\numpy_dataset\\X_test.npy')

# Defining the CS 
cs_pin = digitalio.DigitalInOut(board.C0)
cs_pin.direction = digitalio.Direction.OUTPUT

# Initialize SPI using pyftdi
spi = SpiController()
spi.configure('ftdi://ftdi:232h:1/1') 

# Set the chip select line high to start communication
cs_pin.value = True

# Open SPI bus
master = spi.get_port(cs=0)
master.set_frequency(1e6)


for data in X:
    print("Predictions: ", Run_tflite.test_datum(data))
    data_bytes = []
    for values in data:
        data_ = []
        for value in values:
            value = value[0]
            byte_data = struct.pack('f', value)
            data_.extend(byte_data)
            #print([ "0x%02x" % b for b in byte_data])
        data_bytes.append(bytearray(data_))

    cs_pin.value = False
    for i in range(data.shape[0]):
        # Transmit the byte array via SPI
        cs_pin.value = False
        master.write(data_bytes[i])
        #master.exchange(data_bytes[i], duplex=True)
        #print([ "0x%02x" % b for b in data_bytes[i]])
        cs_pin.value = True
        time.sleep(1e-4)
        

    time.sleep(0.5)


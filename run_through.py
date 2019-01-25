#Run Through
#Myles Scholz

import loaded_Network as ln
import numpy as np
import matplotlib.pyplot as plt

net = ln.loaded_Network("num_net3.p")

def to_data(np_arr):
    return np.reshape(np_arr, (-1, 1))

def to_img(np_arr):
    return plt.imshow(np.reshape(np_arr, (28, 28)))

def result(np_arr):
    return net.feedforward(np_arr)
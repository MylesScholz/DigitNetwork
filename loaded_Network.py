#Loaded Network
#Myles Scholz

import pickle as p
import numpy as np

class loaded_Network(object):
    def __init__(self, file_path):
        #Load weights and biases from a designated .p file
        f = open(file_path, "rb")
        self.weights, self.biases = p.load(f)
        f.close()
        
    def feedforward(self, a):
        #Takes a 28 x 28 numpy array and passes it through the network
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a
    
#Misc Functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))
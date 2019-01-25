#Video Capture
#Myles Scholz

import cv2
from PIL import Image
import PIL.ImageOps
import numpy as np
import matplotlib.pyplot as plt
import run_through as rt
import mnist_loader as ml
tr, te, va = ml.load_data_wrapper()

def run():
    IMAGE_DIR = "C:/Users/Owner/Desktop/Python/DigitNetwork/images/"
    cap = cv2.VideoCapture(0)

    if input("Capture (y/n)?:") == "y": #Capture on 'y'
        ret, frame = cap.read()
        cv2.imwrite(IMAGE_DIR + "c1.png", frame)

    image = Image.open(IMAGE_DIR + "c1.png").convert('L')
    invertedImage = PIL.ImageOps.invert(image)
    resizedImage = invertedImage.resize((28,28))
    resizedImage.save(IMAGE_DIR + "c1e.png") 
    conImage = change_contrast(resizedImage, 100)
    array = np.array(conImage)

    data_in = rt.to_data(array)
    data_in = sigmoid(data_in)
    plt.plot(rt.result(data_in) / 1.0)
    cap.release()
    
#Misc Functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def change_contrast(img, level):
    factor = (259 * (level + 255)) / (255 * (259 - level))
    def contrast(c):
        return 128 + factor * (c - 128)
    return img.point(contrast)

from PIL import Image
import numpy as np

IMAGE_DIR = "C:/Users/Owner/Desktop/Python/images/"
image = Image.open(IMAGE_DIR + "download.png").convert('L')
resizedImage = image.resize((100,100))
resizedImage.save(IMAGE_DIR + "newimagedata.png") 
array = np.array(resizedImage)

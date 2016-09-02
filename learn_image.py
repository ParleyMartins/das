import numpy as np
import cv2
from matplotlib import pyplot as plt

def detect(image):
    height, width, depth = image.shape
    classifier = cv2.CascadeClassifier('Detector_XML/haarcascade_frontalface_alt.xml')
    DOWNSCALE = 4
    minisize = (int(width/DOWNSCALE), int(height/DOWNSCALE))
    # smallimg = cv2.resize(image, minisize)
    faces = classifier.detectMultiScale(image)
    print("Foram encontradas %s faces" % len(faces))


img = cv2.imread('pics/pessoas.jpg', cv2.IMREAD_COLOR)
detect(img)

img = img[:, :, ::-1]

plt.imshow(img, interpolation = 'bicubic')
plt.title('People')
plt.xticks([]), plt.yticks([])
plt.show()

# print('Choose which input method you want:')
# print('1 - Load local file')
#print('2 - Use webcam')
#print('3 - Download file')
#option = input()

#print(option)

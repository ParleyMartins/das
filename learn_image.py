import numpy as np
import cv2
from matplotlib import pyplot as plt

img = cv2.imread('pics/pessoas.jpg', cv2.IMREAD_GRAYSCALE)
cv2.namedWindow('title of the window')
cv2.imshow('title of the window', img)
key = cv2.waitKey(0) & 0xFF #time in miliseconds!
if key == 27: # this is ESC, but I'll look for a flag with this value
    print('Exit without saving.')
    cv2.destroyAllWindows()
elif key == ord('s'):
    print('Saving and exiting.')
    cv2.imwrite('graypeople.png', img)
    cv2.destroyAllWindows()
else:
    print('Key has a weird value.')

import numpy as np
import cv2
import matplotlib.pyplot as plt

%matplotlib gtk

def detect(frame):
    height, width, depth = frame.shape
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(grayscale, grayscale)
    classifier = cv2.CascadeClassifier("Detector_XML/haarcascade_frontalface_alt.xml")
    DOWNSCALE = 4
    minisize = (width/DOWNSCALE, height/DOWNSCALE)
    miniframe = cv2.resize(frame, minisize)
    faces = classifier.detectMultiScale(miniframe)
    if len(faces) > 0:
        print("Face detected!") # Printing in the console
        for face in faces:
            print(face)
            x, y, w, h = [v*DOWNSCALE for v in face]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 25, 5, 0))
    return frame

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
frame = detect(frame)

cap.release()
cv2.destroyAllWindows()

plt.imshow(frame[:,:,::-1])
plt.title('Faces recognized') #Image title
plt.axis('off')

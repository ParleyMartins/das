import numpy as np
import cv2
import matplotlib.pyplot as plt

%matplotlib gtk

def detect(frame):
    height, width, depth = frame.shape
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.equalizeHist(grayscale, grayscale)
    classifier = cv2.CascadeClassifier("/home/parley/src/das/Detector_XML/haarcascade_frontalface_alt.xml")
    DOWNSCALE = 4
    minisize = (frame.shape[1]/DOWNSCALE, frame.shape[0]/DOWNSCALE)
    miniframe = cv2.resize(frame, minisize)
    faces = classifier.detectMultiScale(miniframe)
    if len(faces) > 0:
        print("Face detected!") # Printing in the console
        for face in faces:
            x, y, w, h = [v*DOWNSCALE for v in face]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 25, 5, 0))
    return frame

cap = cv2.VideoCapture(0)

# while(True):
ret, frame = cap.read()
img = frame.copy()
frame = detect(frame)

cv2.imshow('frame', frame) #frame title
    # if(cv2.waitKey(1) & 0xFF == ord('q')):
    #     break

cap.release()
cv2.destroyAllWindows()

plt.imshow(frame[:,:,::-1])
plt.title('VAmos ver o que da nessa droga') #Image title
plt.axis('off')

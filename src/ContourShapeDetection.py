import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0,cv.CAP_DSHOW)

# Check if the webcam is opened correctly
if not capture.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = capture.read()
    frame = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(gray, 127, 255, 0)
    #Find contour, contours is a np array (x,y) coordinates of boundary points of object
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(frame,contours,-1, (0,255,0),2)
    
    cv.imshow('Input', frame)
    
    if cv.waitKey(20) & 0xFF==ord("d"):
        break





#cleanup
capture.release()
cv.destroyAllWindows()

    

import cv2 as cv
import numpy as np

capture = cv.VideoCapture(0,cv.CAP_DSHOW)

# Check if the webcam is opened correctly
if not capture.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = capture.read()
    img = frame.copy()
    
    #convert image to grayscale
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # convert the grayscale image to binary image
    ret,thresh = cv.threshold(gray,127,255,0)
    
    
    #Find contour, contours is a np array (x,y) coordinates of boundary points of object
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        #calculate moments for each contour
        M = cv.moments(c)
        
        #calculate (x,y) of centroid of contour
        if M["m00"] != 0:
            cX = int(M["m10"]/M["m00"])
            cY = int(M["m01"]/M["m00"])
        else:
            cX,cY = 0,0
        cv.circle(img, (cX,cY),5, (0,0,255), -1)       
        #cv.putText(img, "centroid", (cX - 25, cY - 25),cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2) 

    cv.imshow('Input', img)
    
    if cv.waitKey(20) & 0xFF==ord("d"):
        break

#cleanup
capture.release()
cv.destroyAllWindows()

    

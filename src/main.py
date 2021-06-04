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
    
    # detect circles in the image
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1.2, 100)
    
        # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv.rectangle(frame, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
            
    cv.imshow('Input', frame)
    
    if cv.waitKey(20) & 0xFF==ord("d"):
        break

#cleanup
capture.release()
cv.destroyAllWindows()

    








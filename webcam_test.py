import numpy as np
import cv2 as cv

# Use the default webcam (device index 0)
cap = cv.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    # Rotate the frame by 90 degrees
    frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
    
    # Resize the frame to 16:9 aspect ratio
    height, width = frame.shape[:2]
    new_width = int(height * 16 / 9)
    resized_frame = cv.resize(frame, (new_width, height))

    cv.imshow('frame', resized_frame)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()